/* End-to-end checks against serve_observation_fixture.py, never production footage.
   PLAYWRIGHT_MODULE may select an already installed Playwright package. */
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const base=process.env.OBSERVATION_URL || 'http://127.0.0.1:8283';
const out=path.resolve(process.env.OBSERVATION_OUTPUT || 'runs/observation-ui-v2/acceptance');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true}), context=await browser.newContext({viewport:{width:1440,height:1000}}), page=await context.newPage();
 const errors=[],checks=[];context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));page.on('pageerror',e=>errors.push(e.message));
 const episode=async(id='first')=>(await context.request.get(base+'/api/episodes/'+id)).json();
 async function waitSaved(p=page){await p.waitForFunction(()=>document.querySelector('#autosave-status').textContent.startsWith('Saved locally'));}
 async function load(p=page){await p.goto(base);await p.waitForFunction(()=>document.querySelector('#workspace').getAttribute('aria-busy')==='false');}
 try {
  await load(); const original=await episode();assert.equal(original.review.revision,0,'Start a fresh fixture directory');
  assert.equal(await page.locator('.group-card').count(),2);assert.equal(original.observations.length,3);assert.equal(await page.locator('#finish').isDisabled(),true);
  checks.push('Three original observations become two explicit review groups; completion is initially blocked');
  await page.locator('#seek').fill('0');
  const mutations=await page.evaluate(()=>{
   const overlay=document.querySelector('#overlay'), first=overlay.firstElementChild; const observer=new MutationObserver(()=>{});observer.observe(overlay,{childList:true,subtree:true,attributes:true});
   for(const t of [.02,.04,.06,.08,.1,.12,.14,.16,.18,.2]) {const seek=document.querySelector('#seek');seek.value=t;seek.dispatchEvent(new Event('input'));}
   const records=observer.takeRecords();observer.disconnect();return {same:first===overlay.firstElementChild,mutations:records.length};
  }); assert.equal(mutations.same,true);assert.equal(mutations.mutations,0);
  checks.push('Ten seeks within one sample reuse overlay elements with zero overlay DOM mutations');
  let release, requestSeen;const pending=new Promise(r=>requestSeen=r);
  await page.route('**/api/episodes/first/review',async route=>{requestSeen();await new Promise(r=>release=r);await route.continue();},{times:1});
  await page.locator('#event-label').fill('First correction');await pending;
  assert.equal(await page.locator('#discard').isDisabled(),true,'Do not switch episodes during an in-flight write');
  await page.locator('#event-label').fill('Second correction while saving');release();await waitSaved();
  let saved=await episode();assert(saved.review.events.slice(0,2).every(e=>e.label==='Second correction while saving'));
  assert.deepEqual(saved.observations,original.observations);assert.equal(saved.review.outcome,'unknown');
  await page.reload();await page.waitForFunction(()=>document.querySelector('#event-label').value==='Second correction while saving');
  checks.push('Grouped edits autosave, preserve newer in-flight edits, reopen intact, and never rewrite raw evidence or infer outcome');
  await page.locator('#group-members summary').click();await page.locator('#group-member-list button').first().click();
  await page.locator('#timing-details summary').click();await page.locator('#event-end').fill('1.25');await waitSaved();
  saved=await episode();assert.equal(saved.review.events[0].end_seconds,1.25);assert.equal(saved.review.events[1].end_seconds,3);
  checks.push('Individual timing edits leave other grouped observations untouched');
  const second=await context.newPage();await load(second);
  await page.locator('#event-label').fill('Newer saved correction');await waitSaved();
  await second.locator('#event-label').fill('My conflicting draft');await second.waitForFunction(()=>!document.querySelector('#download-draft').hidden);
  const beforeConflict=await episode();assert(!beforeConflict.review.events.some(e=>e.label==='My conflicting draft'));
  const downloadPromise=second.waitForEvent('download');await second.locator('#download-draft').click();const download=await downloadPromise;
  await download.saveAs(path.join(out,'conflicting-draft.json'));const draft=JSON.parse(fs.readFileSync(path.join(out,'conflicting-draft.json'),'utf8'));
  assert.equal(draft.unsaved,true);assert.equal(draft.editor_draft.label,'My conflicting draft');assert.equal(draft.source_sha256,original.source.sha256);
  await second.locator('#reload-review').click();await second.locator('#discard').click();await waitSaved(second);await second.close();
  checks.push('Stale tabs cannot overwrite saved reviews; conflicting drafts remain exportable and reload requires explicit discard');
  await page.route('**/api/episodes/first/review',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Simulated temporary outage'})}),{times:1});
  await page.locator('#event-label').fill('Retryable correction');await page.waitForFunction(()=>!document.querySelector('#save').hidden);
  assert.equal(await page.locator('#event-label').inputValue(),'Retryable correction');
  await page.locator('#save').click();await waitSaved();assert((await episode()).review.events.some(e=>e.label==='Retryable correction'));
  checks.push('A failed save preserves edits and explicit retry persists the same draft');
  let releaseInvalid, invalidRequestSeen;const invalidPending=new Promise(r=>invalidRequestSeen=r);
  await page.route('**/api/episodes/first/review',async route=>{invalidRequestSeen();await new Promise(r=>releaseInvalid=r);await route.continue();},{times:1});
  await page.locator('#event-label').fill('Valid edit before invalid draft');await invalidPending;
  await page.locator('#event-label').fill('');releaseInvalid();
  await page.waitForFunction(()=>document.querySelector('#autosave-status').textContent==='Check the description and timing');
  assert.equal(await page.locator('#download').isDisabled(),true);
  await page.locator('#next').click();await page.locator('#stay').click();assert.equal(await page.locator('#event-label').inputValue(),'');
  await page.locator('#event-label').fill('Retryable correction');await waitSaved();
  checks.push('An older in-flight save cannot mark a newer invalid draft as saved');
  await page.locator('#event-label').fill('');assert.equal(await page.locator('#finish').isDisabled(),true);
  await page.locator('#next').click();await page.locator('#stay').click();assert.equal(await page.locator('#event-label').inputValue(),'');
  await page.locator('#event-label').fill('Retryable correction');await waitSaved();
  checks.push('Invalid drafts block completion and survive attempted navigation');
  while(await page.locator('.group-card.pending').count()) {await page.locator('.group-card.pending').first().click();await page.locator('#approve').click();}
  await waitSaved();
  await page.route('**/api/episodes/first/review',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Simulated completion failure'})}),{times:1});
  await page.locator('#finish').click();await page.waitForFunction(()=>!document.querySelector('#save').hidden);
  assert.equal(await page.locator('#finish').textContent(),'Finish review');assert.equal((await episode()).review.status,'in_review');
  await page.locator('#save').click();await waitSaved();assert.equal((await episode()).review.status,'reviewed');assert.equal((await episode()).review.outcome,'unknown');
  assert.equal(await page.locator('#completion-heading').textContent(),'Review complete');
  checks.push('Failed completion remains visibly unsaved and retry confirms completion');
  const exportPromise=page.waitForEvent('download');await page.locator('#download').click();const exp=await exportPromise;await exp.saveAs(path.join(out,'saved-review.json'));
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out,'saved-review.json'),'utf8')).review,(await episode()).review);
  checks.push('Confirm-and-next resolves the queue; finishing and export preserve the saved revision and unknown outcome');
  await page.locator('#next').click();await page.waitForFunction(()=>document.querySelector('#episode-count').textContent.startsWith('2 /'));
  await page.locator('#add-event').click();await page.locator('#remove-manual').click();assert.equal((await episode('second')).review.events.length,3);
  await page.locator('#add-event').click();await page.locator('#event-label').fill('Manual test event');await page.locator('#approve').click();await waitSaved();
  assert((await episode('second')).review.events.some(e=>e.observation_id===null && e.label==='Manual test event'));
  checks.push('Manual events can be discarded before saving or added without detector evidence');
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  for(const size of [{width:390,height:844},{width:320,height:740},{width:844,height:390}]){
   await page.setViewportSize(size);await page.evaluate(()=>window.scrollTo(0,0));
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
   const mobile=size.width<=820;await page.locator('.group-card').first().click();
   if(mobile){
    assert.equal(await page.locator('#review-editor').evaluate(n=>n.matches(':modal')),true);
    await page.locator('#replay').click();await page.waitForFunction(()=>!document.querySelector('#video').paused);
    await page.locator('#close-editor').click();await page.locator('.group-card').first().click();
    await page.waitForFunction(()=>{const top=document.querySelector('#video-stage').getBoundingClientRect().top;return top>=0 && top<=12});
    assert.equal(await page.evaluate(()=>document.documentElement.style.overflow),'hidden');
   }
   const undersized=await page.locator('#review-editor button:visible,#review-editor input:visible').evaluateAll(nodes=>nodes.filter(n=>n.getBoundingClientRect().height<43.5).map(n=>n.id));assert.deepEqual(undersized,[]);
   await page.screenshot({path:path.join(out,`${size.width}x${size.height}.png`)});
   if(mobile){await page.locator('#close-editor').click();assert.equal(await page.evaluate(()=>document.documentElement.style.overflow),'');}
  }
  checks.push('390px, 320px and landscape layouts have no horizontal overflow; mobile sheets replay footage and expose 44px controls');
  await page.setViewportSize({width:390,height:844});await page.locator('.group-card.pending').first().click();
  await page.keyboard.press('Escape');await page.waitForFunction(()=>document.documentElement.style.overflow==='');
  await page.locator('.group-card.pending').first().click();
  await page.setViewportSize({width:844,height:390});
  await page.waitForFunction(()=>!document.querySelector('#review-editor').matches(':modal') && document.documentElement.style.overflow==='');
  await page.setViewportSize({width:390,height:844});await page.locator('.group-card.pending').first().click();
  while(await page.locator('.group-card.pending').count())await page.locator('#approve').click();
  await page.waitForFunction(()=>!document.querySelector('#review-editor').open && document.documentElement.style.overflow==='');
  assert.equal(await page.locator('#finish').evaluate(n=>{const r=n.getBoundingClientRect();return r.top>=0 && r.bottom<=innerHeight}),true);await waitSaved();
  checks.push('Closing, reopening and rotating release the mobile scroll lock; finishing the queue brings Finish review into view');
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'receipt.json'),JSON.stringify({scope:'Synthetic workflow tests; not model accuracy',checks,errors,overlayMutationsWithinSample:mutations.mutations},null,2)+'\n');console.log(JSON.stringify({checks,errors},null,2));
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
