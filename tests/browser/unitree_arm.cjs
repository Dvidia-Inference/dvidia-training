const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.DVIDIA_PLAYWRIGHT_MODULE || '../../viewer/node_modules/playwright');
const fixture = side => JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../benchmarks/unitree-r1-7a-v01/viewer/benchmark-'+side+'.json')));

async function verify(page, side) {
  await page.getByRole('button',{name:side==='left'?'Left':'Right',exact:true}).click();
  await page.waitForFunction(expected => document.querySelector('#receipt').getAttribute('href')==='benchmark-'+expected+'.json' && document.querySelectorAll('#joints input').length===7 && !document.querySelector('#status').textContent,side);
  const controls=page.getByRole('slider');
  const report=fixture(side);
  const start=await page.locator('#position').textContent();
  for (const name of ['zero','mixed_a','mixed_b']) {
    const pose=report.cases.find(r=>r.case===name);
    await controls.evaluateAll((sliders,angles) => sliders.forEach((slider,index)=>{
      slider.value=angles[index];
      slider.dispatchEvent(new Event('input',{bubbles:true}));
    }),pose.joint_positions_rad);
    const text=await page.locator('#position').textContent();
    const values=[...text.matchAll(/[XYZ] (-?\d+\.\d+)/g)].map(match=>Number(match[1]));
    assert.equal(values.length,3);
    for (let i=0;i<3;i++) assert.ok(Math.abs(values[i]-pose.base_from_link7[i][3])<.001, side+' '+name+' '+text);
  }
  await page.getByRole('button',{name:'Reset pose'}).click();
  assert.equal(await page.locator('#position').textContent(),start);
}

(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disk-cache-size=2048']});
 try {
   const page=await browser.newPage({viewport:{width:1080,height:780}});
   const errors=[]; page.on('pageerror',error=>errors.push(String(error)));
   await page.goto(process.argv[2],{waitUntil:'domcontentloaded'});
   await verify(page,'right');
   if (process.env.DVIDIA_SCREENSHOT) await page.screenshot({path:process.env.DVIDIA_SCREENSHOT});
   await page.setViewportSize({width:390,height:780});
   await verify(page,'left');
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   assert.deepEqual(errors,[]);
   console.log('Both sides loaded. Six displayed wrist poses matched the offline benchmark within 1 mm, including slider rounding. Reset and mobile layout passed.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
