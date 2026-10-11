/* Recorded native task replay smoke checks. No learned policy or physical robot claim. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const browserTypes=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.REFERENCE_TASK_REPLAY_URL||'http://127.0.0.1:8294/';
const out=path.resolve(process.env.REFERENCE_TASK_REPLAY_OUTPUT||'runs/reference-task-browser');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const kind=process.env.PLAYWRIGHT_BROWSER||'chromium';
 const browser=await browserTypes[kind].launch({headless:true,...(kind==='chromium'?{args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']}:{}),...(process.env.PLAYWRIGHT_EXECUTABLE_PATH?{executablePath:process.env.PLAYWRIGHT_EXECUTABLE_PATH}:{}),...(process.env.PLAYWRIGHT_CHANNEL?{channel:process.env.PLAYWRIGHT_CHANNEL}:{})});
 const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
 const errors=[],remote=[],checks=[],requests=[];
 page.on('pageerror',error=>errors.push(error.message));
 page.on('request',request=>{requests.push(request.url());if(new URL(request.url()).origin!==new URL(base).origin&&!request.url().startsWith('data:'))remote.push(request.url());});
 const json=async href=>{const response=await context.request.get(new URL(href,base).href);assert.ok(response.ok(),href);return response.json();};
 const seek=async frame=>{await page.locator('#timeline').evaluate((node,frame)=>{node.value=String(frame);node.dispatchEvent(new Event('input',{bubbles:true}));},frame);await page.waitForFunction(frame=>document.querySelector('#status').textContent.includes(`Recorded frame ${frame+1}/`),frame);};
 try{
  await page.goto(base,{waitUntil:'domcontentloaded',timeout:60000});
  await page.waitForFunction(()=>document.getElementById('stage')?.dataset.ready==='true',{},{timeout:60000});
  const [scene,motion,report]=await Promise.all([json('scene.json'),json('motion.json'),json('receipt/report.json')]);
  const block=scene.bodies.find(b=>b.name==='task_block'),target=scene.bodies.find(b=>b.name==='task_target_frame');
  assert.ok(block);assert.ok(target);
  for(const name of ['task_block_geom','task_table','task_target'])assert.ok(scene.geoms.some(g=>g.name===name&&g.type==='box'));
  assert.equal(await page.locator('#case').inputValue(),motion.cases[0].name);
  assert.equal(await page.locator('#joint-readouts .joint').count(),6);
  assert.equal(await page.locator('#task-evidence tbody tr').count(),motion.cases.length);
  const copy=await page.locator('body').innerText();
  assert.match(copy,/free rigid block and table/i);assert.match(copy,/does not rerun physics/);assert.match(copy,/excerpt is not a success-rate denominator/);
  assert.ok(copy.includes(`complete benchmark contains ${report.attempted} planned attempts`));
  checks.push('Native block, table and target geometry; six joint readouts; authored-controller scope and full source denominator');
  await page.screenshot({path:path.join(out,'desktop-initial.png'),fullPage:true});
  await page.locator('#canvas').screenshot({path:path.join(out,'initial-view.png')});
  const packed=await context.request.get(new URL(motion.motion_binary.compressed_href,base).href);
  assert.ok(packed.ok());
  const binary=zlib.gunzipSync(await packed.body());assert.equal(binary.length,motion.motion_binary.bytes);
  const rawById=new Map();
  for(const attempt of motion.cases){
   const raw=await json(attempt.source_result_href);rawById.set(attempt.attempt_id,raw);
   assert.equal(raw.valid,true);assert.equal(raw.success,attempt.recorded_success);
   assert.equal(attempt.frame_count,raw.trajectory.length);
   assert.equal(attempt.rows.at(-1).terminal_command_carried_forward,true);
   assert.equal(raw.trajectory.at(-1).issued_joint_targets_rad,null);
   for(let frame=0;frame<attempt.frame_count;frame++)for(let axis=0;axis<3;axis++){
    const offset=attempt.pose_byte_offset+(frame*motion.frame_stride_floats+block.id*7+axis)*4;
    assert.ok(Math.abs(binary.readFloatLE(offset)-raw.trajectory[frame].object_position_m[axis])<1e-6,`${attempt.attempt_id} block frame ${frame}`);
   }
   const targetOffset=attempt.pose_byte_offset+target.id*7*4;
   for(let axis=0;axis<2;axis++)assert.ok(Math.abs(binary.readFloatLE(targetOffset+axis*4)-raw.config.target_position_m[axis])<1e-6);
   await page.locator('#case').selectOption(attempt.name);await seek(attempt.frame_count-1);
   assert.equal(Number(await page.locator('#timeline').getAttribute('max')),attempt.frame_count-1);
   assert.equal(await page.locator('#step option').count(),1);
   for(let i=0;i<6;i++){
    const value=attempt.rows.at(-1).joint_position_rad[i]*180/Math.PI;
    assert.equal(await page.locator('#joint-'+i).textContent(),`${value>=0?'+':''}${value.toFixed(1)}°`);
   }
  }
  checks.push('Every selected outcome matches its sealed source; every block pose matches the recorded native state; all controls replay to their terminal row');
  const demonstrated=motion.cases.find(c=>c.control==='placement'&&c.recorded_success)||motion.cases.find(c=>c.control==='placement');
  assert.ok(demonstrated);
  const raw=rawById.get(demonstrated.attempt_id);
  await page.locator('#case').selectOption(demonstrated.name);
  const transfer=raw.trajectory.map((row,index)=>({row,index})).filter(({row,index})=>raw.actions[index]?.phase==='transfer'&&row.object_position_m[2]-raw.initial_object_position_m[2]>.04);
  const carry=transfer.length?transfer[Math.floor(transfer.length/2)].index:raw.trajectory.reduce((best,row,index)=>row.object_position_m[2]>raw.trajectory[best].object_position_m[2]?index:best,0);
  await seek(carry);await page.locator('#canvas').screenshot({path:path.join(out,'carry-view.png')});
  await page.screenshot({path:path.join(out,'desktop-carry.png'),fullPage:true});
  if(demonstrated.recorded_success){
   assert.ok(transfer.length>0,'A completed placement must include observed elevated transfer');
   const finish=raw.trajectory.at(-1).object_position_m;
   assert.ok(Math.hypot(...finish.map((value,index)=>value-raw.target_position_m[index]))<=raw.config.goal_position_tolerance_m+1e-6);
  }
  await seek(demonstrated.frame_count-1);await page.locator('#canvas').screenshot({path:path.join(out,'final-view.png')});
  checks.push('Initial, observed carry and terminal target views captured from actual state; completed source labels require measured lifted transfer and final target tolerance');
  await seek(carry);await page.locator('#collision').click();assert.equal(await page.locator('#mode-label').textContent(),'Native collision proxies');await page.locator('#collision').click();
  await page.locator('#play').click();await page.waitForTimeout(100);await page.locator('#play').click();
  await page.waitForTimeout(100);const renders=await page.locator('#canvas').getAttribute('data-render-count');await page.waitForTimeout(180);assert.equal(await page.locator('#canvas').getAttribute('data-render-count'),renders);
  for(const size of [{width:390,height:844},{width:320,height:740}]){
   await page.setViewportSize(size);await page.waitForTimeout(120);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   const small=await page.locator('button:visible,select:visible,input:visible').evaluateAll(nodes=>nodes.filter(n=>n.getBoundingClientRect().height<43.5).map(n=>n.id));assert.deepEqual(small,[]);
   await page.screenshot({path:path.join(out,`${size.width}x${size.height}-carry.png`),fullPage:true});
  }
  checks.push('Recorded playback, collision mode and idle rendering; 390 px and 320 px layouts keep 44 px controls without horizontal overflow');
  assert.ok(requests.some(url=>url.endsWith('/geometry.bin.gz')));assert.ok(requests.some(url=>url.endsWith('/motion.bin.gz')));
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  const result={scope:'Browser recorded-task replay acceptance; no learned-policy or physical qualification',demonstrated_attempt_id:demonstrated.attempt_id,recorded_success:demonstrated.recorded_success,carry_frame:carry,carry_seconds:raw.trajectory[carry].t_seconds,rendered_attempts:motion.cases.length,full_benchmark_attempted:report.attempted,checks,errors,remote};
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
