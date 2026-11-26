/* Run against a dedicated local Training Studio started with the verified PiPER models. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require(process.env.DVIDIA_PLAYWRIGHT_MODULE || path.join(__dirname, '../../viewer/node_modules/playwright'));

async function ready(page) {
  await page.waitForFunction(() => !document.querySelector('#remember').disabled || !document.querySelector('#error').hidden);
  assert.equal(await page.locator('#error').isVisible(), false, await page.locator('#error').textContent());
}

async function tidy(page) {
  await page.getByRole('button', {name:'Tidy this table'}).click();
  await page.waitForFunction(() => !document.querySelector('#outcome').hidden || !document.querySelector('#error').hidden,
    undefined, {timeout:120000});
  assert.equal(await page.locator('#error').isVisible(), false, await page.locator('#error').textContent());
  return JSON.parse(await page.locator('#record').textContent());
}

async function run(url, output) {
  assert(['localhost', '127.0.0.1'].includes(new URL(url).hostname), 'Use a dedicated loopback Studio.');
  const browser = await chromium.launch({executablePath:process.env.DVIDIA_BROWSER_EXECUTABLE || undefined});
  try {
    await fs.mkdir(output, {recursive:true});
    const page = await browser.newPage({viewport:{width:1120,height:1150}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    await page.getByRole('link', {name:'Remember and tidy a table'}).click();
    await page.waitForFunction(() => !document.querySelector('#preset').disabled);
    await page.locator('#preset').selectOption('home');
    await page.getByLabel('Allow A and B to trade places').check();
    await page.getByRole('button', {name:'Remember this table'}).click();
    await ready(page);
    await page.waitForFunction(() => document.querySelector('#ref-image').naturalWidth === 320);
    assert.match(await page.locator('#reference-text').textContent(), /A and B may trade places/);
    await page.locator('#preset').selectOption('swap');
    const equivalent = await tidy(page);
    assert.equal(equivalent.result.status, 'tidy');
    assert.equal(equivalent.result.time_to_first_move_ms, null);
    assert.equal(equivalent.result.moves.length, 0);
    await page.locator('#preset').selectOption('buffer');
    const shuffled = await tidy(page);
    assert(shuffled.result.time_to_first_move_ms > 0, JSON.stringify(shuffled.result));
    assert(shuffled.result.moves.length > 0);
    assert.equal(shuffled.reference_id, equivalent.reference_id);
    assert.equal(shuffled.result.success, shuffled.result.status === 'completed');
    await page.waitForFunction(() => ['first-image','run-image'].every(id => document.getElementById(id).naturalWidth === 320));
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#outcome').hidden);
    assert.match(await page.locator('#record').textContent(), new RegExp(shuffled.id));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    assert.deepEqual(errors, []);
    const result = {browser:browser.version(), equivalent_status:equivalent.result.status,
      equivalent_moves:equivalent.result.moves.length, shuffled_status:shuffled.result.status,
      shuffled_moves:shuffled.result.moves.length, shuffled_first_motion_ms:shuffled.result.time_to_first_move_ms,
      remembered_frame_width:await page.locator('#ref-image').evaluate(image => image.naturalWidth),
      restored_receipt:true, mobile_no_overflow:true, page_errors:errors};
    await fs.writeFile(path.join(output,'checks.json'), JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
}
run(...process.argv.slice(2)).catch(error => {console.error(error);process.exitCode = 1;});
