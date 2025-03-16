from html2text import html2text
from playwright.sync_api import sync_playwright


def navigate_to_url(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # save screenshot
        page.screenshot(path="page.png")

        # save as PDF
        page.pdf(path="page.pdf")

        # save as html
        html_content = page.content()
        with open("page.html", "w", encoding="utf-8") as f:
            f.write(html_content)

        with open("page.md", "w", encoding="utf-8") as f:
            markdown_content = html2text(html_content)
            f.write(markdown_content)
        browser.close()


if __name__ == "__main__":
    # uv pip install playwright
    # playwright install
    # playwright install-deps
    navigate_to_url("https://www.example.com")
