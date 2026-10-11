/* Read-only browser acceptance on the authored motion benchmark, no private footage. */
const browserTypes=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const base=process.env.MOTION_URL||'http://127.0.0.1:8286';
const out=path.resolve(process.env.MOTION_OUTPUT||'runs/motion-ui-20261010');fs.mkdirSync(out,{recursive:true});
(async()=>{
 // DOM assertions need Playwright's injected evaluator. This affects only the
 // isolated test browser; the server's production CSP remains unchanged.
 const browser=await browserTypes[process.env.PLAYWRIGHT_BROWSER||'chromium'].launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL?{channel:process.env.PLAYWRIGHT_CHANNEL}:{}),...(process.env.PLAYWRIGHT_EXECUTABLE_PATH?{executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH}:{})}),context=await browser.newContext({viewport:{width:1440,height:1000},bypassCSP:true}),page=await context.newPage();
 const errors=[],checks=[];page.on('pageerror',error=>errors.push(error.message));
 async function seek(t){await page.locator('#timeline').fill(String(t));await page.waitForFunction(t=>Math.abs(document.querySelector('#video').currentTime-t)<.002,t);}
 async function select(id){await page.locator('#episode').selectOption(id);await page.waitForFunction(id=>document.querySelector('#video').src.endsWith('/'+id+'/video')&&document.querySelector('#video').readyState>=1,id);}
 try{
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#video').readyState>=1&&document.querySelector('#score-rows').children.length===3);
  assert.equal(await page.locator('#episode').inputValue(),'constant_velocity');assert.equal(await page.locator('#error').isVisible(),false);
  await seek(.02);assert.match(await page.locator('#state').textContent(),/warming up/);assert.equal(await page.locator('#speed').textContent(),'—');
  await seek(.051);const first=(await(await context.request.get(base+'/api/episodes/constant_velocity')).json()).run;
  const row=first.rows.filter(r=>r.acquired_time_seconds<=.051).at(-1);assert.equal(await page.locator('#speed').textContent(),row.speed_pixels_per_second.toFixed(1));assert.match(await page.locator('#age').textContent(),/17\.7 ms/);
  checks.push('Playhead uses only the latest past observation; no velocity appears before two actual samples');
  await page.getByRole('button',{name:'500 ms',exact:true}).click();assert.equal(await page.getByRole('button',{name:'500 ms',exact:true}).getAttribute('aria-pressed'),'true');
  await page.locator('#envelope').uncheck();assert.equal(await page.locator('#envelope').isChecked(),false);await page.locator('#envelope').check();
  assert.match(await page.locator('#score-rows').textContent(),/500 ms/);assert.match(await page.locator('#dimensions').textContent(),/640 × 360/);
  checks.push('Forecast horizon and heuristic envelope controls work; dimensions and image units are explicit');
  const downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Export run receipt'}).click();const download=await downloadPromise;await download.saveAs(path.join(out,'receipt-export.json'));assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out,'receipt-export.json'),'utf8')),first);
  checks.push('Downloaded receipt is the exact served evidence rather than UI-generated predictions');
  await select('brief_occlusion');await seek(2.5);assert.match(await page.locator('#state').textContent(),/Tracking lost/);assert.equal(await page.locator('#speed').textContent(),'—');assert.equal(await page.locator('#anchor').textContent(),'—');
  await select('low_texture');await seek(.5);assert.match(await page.locator('#state').textContent(),/Tracking lost/);assert.match(await page.locator('#score-rows').textContent(),/0 \/ 117/);
  checks.push('Occlusion and low texture visibly abstain instead of displaying invented motion');
  await select('constant_velocity');await seek(1.2);await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  for(const size of [{width:390,height:844},{width:320,height:740},{width:844,height:390}]){
   await page.setViewportSize(size);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   const undersized=await page.locator('button:visible,select:visible').evaluateAll(nodes=>nodes.filter(n=>n.getBoundingClientRect().height<43.5).map(n=>n.id));assert.deepEqual(undersized,[]);
   await page.screenshot({path:path.join(out,`${size.width}x${size.height}.png`),fullPage:true});
  }
  checks.push('Desktop, 390px, 320px and landscape layouts have no page overflow and retain 44px controls');
  // Deliberately remove later API observations in a separate UI-only replay to
  // exercise expiration without changing any served source evidence on disk.
  await page.route('**/api/episodes/constant_velocity',async route=>{const data=await(await route.fetch()).json();data.run.rows=data.run.rows.slice(0,2);await route.fulfill({json:data})});
  await select('stationary');await select('constant_velocity');await seek(.4);assert.match(await page.locator('#state').textContent(),/expired/);assert.equal(await page.locator('#speed').textContent(),'—');checks.push('An expired observation withholds forecasts at later playhead times');
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({scope:'Authored synthetic replay workflow checks; not model accuracy or live latency',checks,errors},null,2)+'\n');console.log(JSON.stringify({checks,errors},null,2));
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
