import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

PRODUCT_URL = (
    "https://thedermaco.com/products/"
    "2-sali-cinamide-anti-acne-face-wash-with-2-salicylic-acid-"
    "2-niacinamide-150-ml-pack-of-2"
)

OUTPUT_CSV = "thedermaco_all_reviews.csv"
MAX_LOAD_MORE_CLICKS = 200
WAIT_AFTER_CLICK_MS = 1800
MAX_NO_PROGRESS = 3


def extract_reviews(page):
    found = {}
    cards = page.locator(".jdgm-rev")

    for i in range(cards.count()):
        try:
            row = cards.nth(i).evaluate("""el => {
                const txt = s => {
                    const n = el.querySelector(s);
                    return n ? n.innerText.trim() : "";
                };
                const rating = el.querySelector(".jdgm-rev__rating");
                const date = el.querySelector(".jdgm-rev__timestamp");
                const body = el.querySelector(".jdgm-rev__body");

                return {
                    review_id: el.getAttribute("data-review-id") || "",
                    product_name: el.getAttribute("data-product-title") || "",
                    product_url: el.getAttribute("data-product-url") || "",
                    reviewer_name: txt(".jdgm-rev__author"),
                    rating: rating ? rating.getAttribute("data-score") || "" : "",
                    review_title: txt(".jdgm-rev__title"),
                    review_text: body ? body.innerText
                        .replace(/\\s*Read more\\s*/gi, "").trim() : "",
                    review_date: date
                        ? date.getAttribute("datetime") || date.innerText.trim() : "",
                    verified_buyer: el.getAttribute("data-verified-buyer") || "",
                    helpful_votes: el.getAttribute("data-thumb-up-count") || "",
                    not_helpful_votes: el.getAttribute("data-thumb-down-count") || ""
                };
            }""")

            rid = row["review_id"]
            key = rid if rid else (
                row["reviewer_name"], row["review_date"], row["review_text"]
            )
            found[key] = row
        except Exception as exc:
            print(f"Could not parse review {i}: {exc}")

    return found


def click_load_more(page):
    # Try known Judge.me selectors first.
    selectors = [
        ".jdgm-rev-widg__load-more",
        ".jdgm-all-reviews__load-more",
        ".jdgm-load-more",
        ".jdgm-rev-widg__load-more-btn",
        "button[class*='load-more']",
        "a[class*='load-more']"
    ]

    for selector in selectors:
        loc = page.locator(selector).last
        try:
            if loc.count() and loc.is_visible() and loc.is_enabled():
                loc.scroll_into_view_if_needed()
                loc.click(timeout=5000)
                return True, selector
        except Exception:
            pass

    # The live page exposes a button labelled "Load More".
    try:
        loc = page.get_by_role("button", name=re.compile(r"load\\s*more", re.I)).last
        if loc.count() and loc.is_visible() and loc.is_enabled():
            loc.scroll_into_view_if_needed()
            loc.click(timeout=5000)
            return True, "button role/name"
    except Exception:
        pass

    # Fallback for a div/span styled as a button.
    try:
        loc = page.get_by_text(re.compile(r"^\\s*load\\s*more\\s*$", re.I)).last
        if loc.count() and loc.is_visible():
            loc.scroll_into_view_if_needed()
            loc.click(timeout=5000)
            return True, "visible Load More text"
    except Exception:
        pass

    # Last fallback: click a visible DOM element whose text contains Load More.
    try:
        clicked = page.evaluate("""() => {
            const els = [...document.querySelectorAll('button, a, [role="button"], div, span')];
            const el = els.reverse().find(e =>
                e.innerText && /^\\s*load\\s*more\\s*$/i.test(e.innerText) &&
                e.getClientRects().length
            );
            if (!el) return false;
            el.click();
            return true;
        }""")
        if clicked:
            return True, "DOM text fallback"
    except Exception:
        pass

    return False, "No visible Load More control"


def main():
    reviews = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1100},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        try:
            response = page.goto(
                PRODUCT_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )
            print("Product page HTTP status:", response.status if response else "unknown")
            page.wait_for_timeout(5000)

            try:
                page.locator(".jdgm-rev").first.wait_for(
                    state="attached", timeout=30000
                )
            except PlaywrightTimeoutError:
                print("Judge.me cards not detected. Saving page diagnostics.")
                page.screenshot(path="scraper_debug.png", full_page=True)
                print("Page title:", page.title())
                print("Review card count:", page.locator(".jdgm-rev").count())
                return

            no_progress = 0

            for click_no in range(MAX_LOAD_MORE_CLICKS + 1):
                # Expand review bodies where possible
                expanders = page.locator(
                    ".jdgm-rev__body-read-more, "
                    ".jdgm-rev__read-more, "
                    ".jdgm-rev__body a[aria-label='Read more']"
                )
                for j in range(expanders.count()):
                    try:
                        e = expanders.nth(j)
                        if e.is_visible():
                            e.click(timeout=1000)
                    except Exception:
                        pass

                before = len(reviews)
                reviews.update(extract_reviews(page))
                after = len(reviews)
                print(
                    f"Pass {click_no + 1}: "
                    f"{page.locator('.jdgm-rev').count()} cards in DOM; "
                    f"{after} unique reviews collected."
                )

                if click_no == MAX_LOAD_MORE_CLICKS:
                    print("Reached MAX_LOAD_MORE_CLICKS safety limit.")
                    break

                clicked, method = click_load_more(page)
                if not clicked:
                    print(method)
                    break

                print("Clicked Load More using:", method)

                # Wait for newly appended/replaced review cards.
                try:
                    page.wait_for_function(
                        "(oldCount) => document.querySelectorAll('.jdgm-rev').length > oldCount",
                        arg=page.locator(".jdgm-rev").count(),
                        timeout=12000
                    )
                except PlaywrightTimeoutError:
                    page.wait_for_timeout(WAIT_AFTER_CLICK_MS)

                page.wait_for_timeout(500)
                reviews.update(extract_reviews(page))

                if len(reviews) <= before:
                    no_progress += 1
                    print(f"No new unique reviews after click ({no_progress}/{MAX_NO_PROGRESS}).")
                else:
                    no_progress = 0

                if no_progress >= MAX_NO_PROGRESS:
                    print("Stopping because repeated clicks did not add new reviews.")
                    break

            # Final extraction after pagination completes
            reviews.update(extract_reviews(page))

        finally:
            browser.close()

    columns = [
        "review_id", "product_name", "reviewer_name", "rating",
        "review_title", "review_text", "review_date", "verified_buyer",
        "helpful_votes", "not_helpful_votes", "product_url"
    ]

    df = pd.DataFrame(list(reviews.values()))
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]

    for col in ["rating", "helpful_votes", "not_helpful_votes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\\n========== SCRAPING SUMMARY ==========")
    print("Unique reviews saved:", len(df))
    print("CSV file:", OUTPUT_CSV)
    print("Website displayed total: 162 (as shown when checked)")
    print("Compare the collected count with the website total.")
    if len(df) < 162:
        print(
            "WARNING: Fewer than 162 reviews were collected. "
            "The page may use another pagination/API mechanism; "
            "do not treat this partial CSV as the complete dataset."
        )


if __name__ == "__main__":
    main()
