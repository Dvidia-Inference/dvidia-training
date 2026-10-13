/* Live workstation browser acceptance.
 * Usage: node tests/browser/workstation_ui.cjs URL NEW_OUTPUT_DIRECTORY
 * Set DVIDIA_PLAYWRIGHT_MODULE when Playwright is installed outside this repo.
 * This measures functional interactions, not hardware or rendering performance.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.DVIDIA_PLAYWRIGHT_MODULE||'playwright');
async function pixels(page){
 await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
 const image=await page.locator('#canvas').screenshot();
 return await page.evaluate(async data=>{const im=new Image();im.src='data:image/png;base64,'+data;await im.decode();const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const ctx=c.getContext('2d');ctx.drawImage(im,0,0);const p=ctx.getImageData(0,0,c.width,c.height).data;let wood=0,navy=0;const red=[];for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++){const i=(y*c.width+x)*4,r=p[i],g=p[i+1],b=p[i+2];if(r>120&&r-g>6&&g-b>15)wood++;if(r<100&&g<140&&b<160&&b-r>4)navy++;if(r>150&&g<95&&b<95)red.push([x,y]);}return {wood,navy,width:c.width,height:c.height,red};},image.toString('base64'));
}
async function state(page){return page.evaluate(()=>window.__workstationState);}
(async()=>{
 const [url,output]=process.argv.slice(2);
 assert(url&&output,'Provide the workstation URL and a new output directory.');
 assert(['http:','https:'].includes(new URL(url).protocol),'Use HTTP or HTTPS.');
 await fs.mkdir(output,{recursive:true});
 const file=name=>path.join(output,name);
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
 const errors=[],responses=[];page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)responses.push([r.url(),r.status()])});
 await page.addInitScript(()=>{const Native=window.Worker;window.Worker=class extends Native{constructor(...args){super(...args);this.addEventListener('message',e=>{if(e.data?.state)window.__workstationState=e.data.state;});}};});
 try {
 await page.goto(url,{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>document.querySelector('#stage')?.dataset.ready==='true',{},{timeout:120000});
 await page.waitForFunction(()=>window.__workstationState?.sim_time>.3);
 const before=await state(page);
 await page.locator('#pause').click();await page.waitForFunction(()=>window.__workstationState?.paused===true);
 const stopped=await state(page);await page.waitForTimeout(400);assert.equal((await state(page)).sim_time,stopped.sim_time);
 const px=await pixels(page);assert(px.wood>1000&&px.navy>150,'Actual native scene must render');assert(px.red.length>10,'Translation gizmo must render');
 const sorted=px.red.sort((a,b)=>b[0]-a[0]);const right=sorted.slice(0,Math.min(20,sorted.length));const gx=right.reduce((s,p)=>s+p[0],0)/right.length,gy=right.reduce((s,p)=>s+p[1],0)/right.length;
 const box=await page.locator('#canvas').boundingBox();const start=(await state(page)).target_position;
 await page.mouse.move(box.x+gx,box.y+gy);await page.mouse.down();await page.mouse.move(box.x+gx+35,box.y+gy+7,{steps:12});await page.mouse.up();
 await page.waitForFunction(old=>window.__workstationState.target_position.some((n,i)=>Math.abs(n-old[i])>.001),start,{timeout:10000});
 const dragged=(await state(page)).target_position;
 await page.locator('#pause').click();await page.waitForFunction(()=>window.__workstationState?.paused===false);
 for(const [axis,value]of [['x','370'],['y','-100'],['z','240']])await page.locator('#target-'+axis).fill(value);
 await page.locator('#target-z').blur();await page.waitForTimeout(300);assert.deepEqual(await Promise.all(['x','y','z'].map(a=>page.locator('#target-'+a).inputValue())),['370','-100','240'],'Blurred coordinate drafts survive live states');await page.locator('#apply-target').click();
 await page.waitForFunction(()=>Math.abs(window.__workstationState.target_position[1]+.1)<1e-8&&Math.abs(window.__workstationState.target_position[0]-.37)<1e-8);
 await page.waitForFunction(()=>window.__workstationState.tracking_error_m<.01,{},{timeout:15000});
 const reached=await state(page);assert(reached.qpos.some((n,i)=>Math.abs(n-before.qpos[i])>.01),'Actual joint positions must change');
 await page.locator('#gripper').evaluate(e=>{e.value='20';e.dispatchEvent(new Event('input',{bubbles:true}));});
 await page.waitForFunction(()=>Math.abs(window.__workstationState.jaw-.02)<1e-8);
 await page.waitForFunction(()=>Math.abs(window.__workstationState.actual_jaw-.02)<.002,{},{timeout:10000});
 const prePush=(await state(page)).shoes.shoe_trainer_left.position;
 await page.locator('#push').click();
 await page.waitForFunction(p=>Math.hypot(...window.__workstationState.shoes.shoe_trainer_left.position.map((n,i)=>n-p[i]))>.001,prePush,{timeout:10000});
 const postPush=(await state(page)).shoes.shoe_trainer_left.position;
 await page.locator('#collision').click();assert.equal(await page.locator('#collision').getAttribute('aria-pressed'),'true');await page.locator('#collision').click();
 await page.locator('#gizmo').click();assert.equal(await page.locator('#gizmo').getAttribute('aria-pressed'),'false');await page.locator('#gizmo').click();
 await page.locator('[data-camera=top]').click();await page.locator('[data-camera=front]').click();await page.locator('#home-camera').click();
 const download=page.waitForEvent('download');await page.locator('#receipt').click();const artifact=await download;await artifact.saveAs(file('session.json'));const receipt=JSON.parse(await fs.readFile(file('session.json'),'utf8'));assert(receipt.commands.some(c=>c.type==='target'));assert(receipt.commands.some(c=>c.type==='push'&&c.impulse));assert(receipt.samples.length>0&&receipt.samples.some(s=>s.body_pos.length>3));
 await page.locator('#reset').click();await page.waitForFunction(()=>window.__workstationState.sim_time<1&&Math.abs(window.__workstationState.target_position[1]+.06)<1e-8);
 // A bounded XYZ box does not guarantee the fixed-orientation tool can reach
 // every point. Verify the UI stops saying Approaching for the known high case.
 for(const [axis,value]of [['x','360'],['y','-60'],['z','320']])await page.locator('#target-'+axis).fill(value);
 await page.locator('#apply-target').click();
 await page.waitForFunction(()=>Math.abs(window.__workstationState.target_position[2]-.32)<1e-8);
 const highStart=(await state(page)).sim_time;
 await page.waitForFunction(start=>window.__workstationState.sim_time>=start+4.1&&document.querySelector('#tool-state').dataset.state==='limited',highStart,{timeout:30000});
 const high=await state(page);
 assert(high.tracking_error_m>.005,'The unreached status requires measured error above 5 mm');
 assert.match(await page.locator('#tool-state').innerText(),/Target not reached/);
 const highTarget={target:high.target_position,error_m:high.tracking_error_m,simulated_seconds:high.sim_time-highStart,status:await page.locator('#tool-state').innerText()};
 await page.screenshot({path:file('target-not-reached.png'),fullPage:true});
 await page.locator('#reset').click();
 await page.waitForFunction(()=>window.__workstationState.sim_time<1&&Math.abs(window.__workstationState.target_position[2]-.27)<1e-8);
 await page.waitForFunction(()=>document.querySelector('#tool-state').dataset.state!=='limited');
 await page.waitForTimeout(500);
 await page.locator('#pause').click();await page.waitForFunction(()=>window.__workstationState.paused===true);
 const layouts=[];
 for(const width of[1440,390,320]){await page.setViewportSize({width,height:width===1440?1100:900});await page.locator('#home-camera').click();const p=await pixels(page);assert(p.wood>1000&&p.navy>30);const caption=await page.locator('#scene-caption').boundingBox();const sceneTools=await page.locator('#gizmo').boundingBox();assert(caption.x+caption.width<=sceneTools.x-8,'Scene caption must not run under display controls at '+width);const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);assert.equal(overflow,false,'No horizontal overflow '+width);await page.screenshot({path:file('workstation-'+width+'.png'),fullPage:true});layouts.push({width,pixels:{wood:p.wood,navy:p.navy},overflow});}
 assert.deepEqual(errors,[]);assert.deepEqual(responses,[]);
 const result={passed:true,url,scope:'Headless browser visual and interaction acceptance; not a hardware performance benchmark.',pointerDrag:{start,dragged},targetResponse:{criterion:'Native joints changed and error fell below 10 mm; not a final precision qualification.',error_m:reached.tracking_error_m,tcp:reached.tcp_position},shoeNudge:{before:prePush,after:postPush},pauseStable:true,blurredCoordinateDraftsPreserved:true,highTarget,receiptCommands:receipt.commands.length,receiptSamples:receipt.samples.length,layouts,errors,responses};
 await fs.writeFile(file('acceptance.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
