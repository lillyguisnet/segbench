# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright>=1.50"]
# ///
"""Browser integration checks. Uses a fresh browser profile, never the user's save.

uv run scripts/test_consensus_review.py URL /path/to/chromium
URL may be the artifact preview (preferred) or a local HTTP server.
"""
import json
import sys
from playwright.sync_api import sync_playwright


def main(url, executable):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
        context = browser.new_context(viewport={'width': 1400, 'height': 1050}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function('window.reviewTest && document.querySelector("#loading").hidden')
        get = lambda: page.evaluate('reviewTest.getState()')
        page.evaluate('reviewTest.validate(reviewTest.getState())')
        assert get()['source_sha256']
        assert len(page.locator('#tasks button').all()) == 6
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/tmp/consensus-review-desktop.png')

        # Review a suggestion, undo, redo; no approval was inferred from model support.
        assert not any(o['status']=='accepted' for o in get()['tasks']['logs']['objects'])
        page.click('#nextUncertain')
        page.keyboard.press('Enter')
        assert sum(o['status']=='accepted' for o in get()['tasks']['logs']['objects']) == 1
        page.click('#undo')
        assert not any(o['status']=='accepted' for o in get()['tasks']['logs']['objects'])
        page.click('#redo')
        assert sum(o['status']=='accepted' for o in get()['tasks']['logs']['objects']) == 1

        # Select and move the accepted point in photo coordinates.
        page.click('#fit')
        obj = next(o for o in get()['tasks']['logs']['objects'] if o['status']=='accepted')
        view = page.evaluate('reviewTest.getView()')
        rect = page.locator('#canvas').bounding_box()
        x = rect['x'] + view['tx'] + obj['x']*view['scale']
        y = rect['y'] + view['ty'] + obj['y']*view['scale']
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+12,y+12,steps=5);page.mouse.up()
        changed = next(o for o in get()['tasks']['logs']['objects'] if o['id']==obj['id'])
        assert changed['x'] != obj['x']
        page.click('#undo')
        assert next(o for o in get()['tasks']['logs']['objects'] if o['id']==obj['id'])['x']==obj['x']

        # Add an object in empty photo space, then reject it and undo.
        page.click('[data-mode="add"]')
        view=page.evaluate('reviewTest.getView()');rect=page.locator('#canvas').bounding_box()
        page.mouse.click(rect['x']+view['tx']+10*view['scale'],rect['y']+view['ty']+10*view['scale'])
        added=[o for o in get()['tasks']['logs']['objects'] if o['origin']=='human']
        assert len(added)==1
        page.click('#reject')
        assert next(o for o in get()['tasks']['logs']['objects'] if o['id']==added[0]['id'])['status']=='rejected'
        page.click('#undo')
        assert next(o for o in get()['tasks']['logs']['objects'] if o['id']==added[0]['id'])['status']=='accepted'

        # Label review, notes and full-photo review flag.
        page.click('[data-task="5"]')
        assert page.locator('#labels').is_visible(), 'Dish controls must be discoverable before selecting a dot'
        assert page.locator('[data-label="clean"]').is_disabled()
        assert page.locator('#labels').bounding_box()['y'] < page.locator('#canvas').bounding_box()['y']
        page.click('#nextUncertain')
        page.click('[data-label="dirty"]')
        selected_id=get()['audit'][-1]['id']
        label=lambda: next(o['label'] for o in get()['tasks']['dishes']['objects'] if o['id']==selected_id)
        assert label()=='dirty'
        page.keyboard.press('s')
        assert label()=='unsure'
        page.click('#undo')
        assert label()=='dirty'
        page.click('[data-label="clean"]')
        assert label()=='clean'
        assert page.locator('[data-label="clean"]').get_attribute('aria-pressed')=='true'
        page.screenshot(path='/tmp/consensus-dish-labels-desktop.png')
        page.click('#accept')
        assert any(o['status']=='accepted' and o['label']=='clean' for o in get()['tasks']['dishes']['objects'])
        page.fill('#notes','Check the stacked dishes separately.')
        page.locator('#notes').blur()
        page.check('#reviewed')
        assert get()['tasks']['dishes']['full_photo_checked']

        # Region brush modifies actual pixels, undo restores the exact original grid.
        page.click('[data-task="4"]')
        before=get()['tasks']['road']['mask']
        page.click('[data-mode="paint"]')
        view=page.evaluate('reviewTest.getView()');rect=page.locator('#canvas').bounding_box()
        x=rect['x']+view['tx']+300*view['scale'];y=rect['y']+view['ty']+300*view['scale']
        page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+40,y+20,steps=6);page.mouse.up()
        assert get()['tasks']['road']['mask'] != before
        page.click('#undo')
        assert get()['tasks']['road']['mask'] == before
        page.click('#redo')
        assert get()['tasks']['road']['mask'] != before
        page.evaluate('reviewTest.validate(reviewTest.getState())')
        page.evaluate('reviewTest.save()')
        saved=get()
        page.reload()
        page.wait_for_function('window.reviewTest && document.querySelector("#loading").hidden')
        assert get()['tasks']==saved['tasks'], 'Browser autosave did not survive a reload'

        # Download is a real portable JSON export, including original evidence.
        with page.expect_download() as event:
            page.click('#export')
        export_path=event.value.path()
        with open(export_path) as f:
            exported=json.load(f)
        assert exported['tasks']==saved['tasks']
        assert exported['source_snapshot']['source_sha256']==saved['source_sha256']

        # Reject a mismatched source without overwriting any state.
        bad=json.loads(json.dumps(exported));bad['source_sha256']='wrong'
        page.set_input_files('#file',{'name':'wrong.json','mimeType':'application/json','buffer':json.dumps(bad).encode()})
        page.wait_for_function('document.querySelector("#toast").textContent.includes("different consensus")')
        assert get()['tasks']==saved['tasks']

        # Import into a second browser context restores all tasks.
        other=browser.new_context(viewport={'width':390,'height':844})
        mobile=other.new_page();mobile.on('pageerror',lambda e:errors.append(str(e)))
        mobile.goto(url);mobile.wait_for_function('window.reviewTest && document.querySelector("#loading").hidden')
        mobile.on('dialog',lambda d:d.accept())
        mobile.set_input_files('#file',{'name':'review.json','mimeType':'application/json','buffer':json.dumps(exported).encode()})
        mobile.wait_for_function('reviewTest.getState().audit.some(e=>e.action==="import")')
        assert mobile.evaluate('reviewTest.getState().tasks')==saved['tasks']
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile layout overflows'
        mobile.screenshot(path='/tmp/consensus-review-mobile.png',full_page=True)
        mobile.click('[data-task="5"]')
        assert mobile.locator('#labels').is_visible()
        assert mobile.locator('#labels').bounding_box()['y'] < mobile.locator('#canvas').bounding_box()['y']
        mobile.click('#nextUncertain')
        mobile.locator('#labels').scroll_into_view_if_needed()
        mobile.click('[data-label="dirty"]')
        assert mobile.locator('[data-label="dirty"]').get_attribute('aria-pressed')=='true'
        mobile.screenshot(path='/tmp/consensus-dish-labels-mobile.png',full_page=True)
        mobile.click('[data-task="0"]')

        # Two-finger gestures in Add mode must not accidentally add an object.
        mobile.click('[data-mode="add"]')
        rect=mobile.locator('#canvas').bounding_box()
        count=len(mobile.evaluate('reviewTest.getState().tasks.logs.objects'))
        cdp=other.new_cdp_session(mobile)
        points=[{'x':rect['x']+100,'y':rect['y']+rect['height']/2},
                {'x':rect['x']+220,'y':rect['y']+rect['height']/2}]
        cdp.send('Input.dispatchTouchEvent',{'type':'touchStart','touchPoints':points})
        points[0]['x']-=15;points[1]['x']+=15
        cdp.send('Input.dispatchTouchEvent',{'type':'touchMove','touchPoints':points})
        cdp.send('Input.dispatchTouchEvent',{'type':'touchEnd','touchPoints':[]})
        assert len(mobile.evaluate('reviewTest.getState().tasks.logs.objects'))==count

        # Storage blocked: editing/export still work, and the UI never claims saved.
        blocked=context.new_page()
        blocked.add_init_script("Storage.prototype.setItem = function(){throw new Error('storage blocked')}")
        blocked.goto(url);blocked.wait_for_function('window.reviewTest && document.querySelector("#loading").hidden')
        blocked.evaluate('reviewTest.save()')
        assert 'Not autosaved' in blocked.locator('#saveStatus').inner_text()

        # Another tab cannot silently overwrite newer corrections.
        second=context.new_page();second.goto(url);second.wait_for_function('window.reviewTest')
        page.click('#nextUncertain');page.click('#accept');page.evaluate('reviewTest.save()')
        second.wait_for_function('document.querySelector("#saveStatus").textContent.includes("Another tab")')
        assert not errors, errors
        print('PASS: rendering, move/add/reject/labels, undo/redo, region brush, autosave/reload, JSON export/import, mismatch rejection, mobile layout.')
        browser.close()

if __name__=='__main__':
    main(*sys.argv[1:])
