# /// script
# dependencies = ["playwright>=1.50"]
# ///
"""Smoke-test the point report through its actual artifact URL in a fresh browser."""
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch(executable_path=sys.argv[2],args=['--no-sandbox'],headless=True)
    page=browser.new_page(viewport={'width':1400,'height':1100})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(sys.argv[1]);page.wait_for_selector('#summary tbody tr')
    assert page.locator('#summary tbody tr').count()==16
    assert page.locator('#chart circle').count()==13
    page.check('#specialists')
    assert page.locator('#chart circle').count()==15
    page.screenshot(path='/tmp/point-report.png',full_page=True)
    for task in ('logs','cows','fig','dishes'):
        page.select_option('#task',task)
        for tol in ('tight','medium','loose'):
            page.select_option('#tol',tol)
            page.wait_for_function('document.querySelector("#photoinner img").complete && document.querySelector("#photoinner img").naturalWidth>0')
            assert page.locator('#photoinner svg').count()==1
    page.select_option('#model','rfdetr-seg-2xl')
    assert 'Not attempted' in page.locator('#error').inner_text()
    page.select_option('#task','cows')
    assert page.locator('#error').inner_text()==''
    page.select_option('#task','logs');page.select_option('#model','gemini-pro@medium')
    assert 'Unreadable' in page.locator('#error').inner_text()
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    print('PASS: table, provisional toggle, all tasks and tolerances, photos, partial coverage, unreadable replies, mobile layout.')
    browser.close()
