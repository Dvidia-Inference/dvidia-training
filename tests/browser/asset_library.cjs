/* Offline/native asset viewer acceptance. Run: node this.cjs URL OUTPUT_DIRECTORY */
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.DVIDIA_PLAYWRIGHT_MODULE||'playwright');
async function renderedPixels(page){
  // Wait for the requested draw and the following compositor frame; a DOM
  // readout can update before an on-demand WebGL frame has been presented.
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const image=await page.locator('#canvas').screenshot();
  const counts=await page.evaluate(async encoded=>{
    const img=new Image();img.src='data:image/png;base64,'+encoded;await img.decode();
    const canvas=document.createElement('canvas');canvas.width=img.width;canvas.height=img.height;
    const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);const pixels=ctx.getImageData(0,0,img.width,img.height).data;
    let colored=0,darkNeutral=0;
    for(let y=Math.floor(img.height*.12);y<img.height*.90;y++)for(let x=Math.floor(img.width*.06);x<img.width*.94;x++){
      const i=(y*img.width+x)*4,r=pixels[i],g=pixels[i+1],b=pixels[i+2],range=Math.max(r,g,b)-Math.min(r,g,b);
      if(range>30&&Math.max(r,g,b)<245)colored++;
      if(range<25&&Math.max(r,g,b)<175)darkNeutral++;
    }
    return {colored,darkNeutral,width:img.width,height:img.height};
  },image.toString('base64'));
  assert(counts.colored>50,'Native colored shoe surfaces must occupy visible pixels; got '+JSON.stringify(counts));
  assert(counts.darkNeutral>30,'Arm/sole geometry must occupy visible pixels; got '+JSON.stringify(counts));
  return {image,counts};
}
async function main(){
  const [url,output]=process.argv.slice(2);
  assert(url&&output,'Provide preview URL and screenshot output directory.');
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  const records=[];
  try {
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1000},deviceScaleFactor:1});
      const errors=[],remote=[];
      page.on('pageerror',e=>errors.push(e.message));
      const origin=new URL(url).origin;
      page.on('request',r=>{if(new URL(r.url()).origin!==origin&&!r.url().startsWith('data:'))remote.push(r.url());});
      await page.goto(url,{waitUntil:'networkidle'});
      await page.waitForFunction(()=>document.querySelector('#stage').dataset.ready==='true',{},{timeout:30000});
      assert.equal(await page.locator('#stage').getAttribute('data-selected-shoes'),'6');
      assert(Number(await page.locator('#stage').getAttribute('data-visual-geoms'))>20);
      await renderedPixels(page);
      const published=/\/previews\/(ur5e|piper|so101)\/(?:index\.html)?$/.test(new URL(url).pathname);
      assert.equal(await page.locator('#robot-links').isVisible(),published);
      await page.locator('#pair').selectOption('slipper');
      assert.equal(await page.locator('#stage').getAttribute('data-selected-shoes'),'2');
      assert.match(await page.locator('#command').innerText(),/--pair slipper/);
      const descriptor=await page.locator('#descriptor').getAttribute('href');
      const response=await page.request.get(new URL(descriptor,url).href);
      assert(response.ok());
      assert.deepEqual((await response.json()).shoe_ids,['slipper-left','slipper-right']);
      await page.locator('#collision').click();
      assert.equal(await page.locator('#collision').getAttribute('aria-pressed'),'true');
      await page.locator('#collision').click();
      await page.locator('#timeline').evaluate(e=>{e.value='0';e.dispatchEvent(new Event('input',{bubbles:true}));});
      assert.match(await page.locator('#clock').innerText(),/^0.00/);
      await page.locator('#play').click();
      await page.waitForFunction(()=>Number(document.querySelector('#timeline').value)>2);
      await page.locator('#play').click();
      await page.locator('[data-camera=top]').click();
      await page.locator('[data-camera=home]').click();
      await page.locator('#pair').selectOption('all');
      const beforeOrbit=await renderedPixels(page);
      await page.locator('#canvas').focus();
      await page.keyboard.press('ArrowLeft');
      const afterOrbit=await renderedPixels(page);
      assert.notDeepEqual(afterOrbit.image,beforeOrbit.image,'Keyboard orbit must change rendered geometry');
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
      assert.equal(overflow,false,'No horizontal overflow at '+width);
      assert.deepEqual(errors,[]);
      // HF may serve larger files through its documented public CDN.
      assert(remote.every(u=>new URL(u).host==='us.aws.cdn.hf.co'),'No unexpected network imports');
      await page.screenshot({path:path.join(output,'asset-'+width+'.png'),fullPage:true});
      records.push({width,renderCount:Number(await page.locator('#canvas').getAttribute('data-render-count')),renderedPixels:afterOrbit.counts,errors,remoteDownloads:remote.length,overflow});
      await page.close();
    }
    // Deterministically reproduce a RAF queued before a later play/resume
    // handler. Its timestamp must not read a negative or earlier pose index.
    const stale=await browser.newPage({viewport:{width:900,height:900}});
    const staleErrors=[];
    stale.on('pageerror',error=>staleErrors.push(error.message));
    await stale.addInitScript(()=>{
      const original=window.requestAnimationFrame.bind(window);
      window.__assetTestPastRaf=false;
      window.requestAnimationFrame=callback=>original(timestamp=>callback(window.__assetTestPastRaf?timestamp-10000:timestamp));
    });
    await stale.goto(url,{waitUntil:'networkidle'});
    await stale.waitForFunction(()=>document.querySelector('#stage').dataset.ready==='true');
    await stale.evaluate(()=>{
      const timeline=document.querySelector('#timeline');timeline.value='60';timeline.dispatchEvent(new Event('input',{bubbles:true}));
      window.__assetTestPastRaf=true;document.querySelector('#play').click();
    });
    await stale.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(await stale.locator('#timeline').inputValue(),'60','An earlier RAF timestamp must preserve the resumed frame');
    await stale.locator('#play').click();
    await stale.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(await stale.locator('#timeline').inputValue(),'60','A queued frame after pause must preserve the pose');
    await stale.evaluate(()=>{
      const timeline=document.querySelector('#timeline');timeline.value='0';timeline.dispatchEvent(new Event('input',{bubbles:true}));
      document.querySelector('#play').click();
    });
    await stale.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    assert.equal(await stale.locator('#timeline').inputValue(),'0','An earlier RAF timestamp must never produce a negative frame');
    await stale.evaluate(()=>{window.__assetTestPastRaf=false;});
    await stale.waitForFunction(()=>Number(document.querySelector('#timeline').value)>0);
    await stale.locator('#play').click();
    assert.deepEqual(staleErrors,[]);
    await stale.close();
    const timestampRegression={forcedTimestampOffsetMs:-10000,resumeFrame:60,pausedFrame:60,initialFrame:0,normalPlaybackResumed:true,errors:staleErrors};
    await fs.writeFile(path.join(output,'acceptance.json'),JSON.stringify({url,records,timestampRegression},null,2)+'\n');
    console.log(JSON.stringify({passed:true,records,timestampRegression},null,2));
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
