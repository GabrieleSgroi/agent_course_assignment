import os
import re
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd
from langchain_community.retrievers import WikipediaRetriever
from playwright.sync_api import sync_playwright
from markdownify import markdownify
from smolagents import DuckDuckGoSearchTool


@dataclass(frozen=True)
class PageResult:
    url: str
    source: str
    content: str


class PlaywrightPageVisit:
    def __init__(self,
                 wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] | None = "load"):
        self.wait_until = wait_until

    def __call__(self, url: str) -> str:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until=self.wait_until)
                html = page.content()
                markdown = markdownify(html)
                browser.close()
        except TimeoutError:
            raise TimeoutError(f"Impossible to load the page {url}")
        return markdown


class InternetSearch:
    def __init__(self,
                 visit_tool: Callable = PlaywrightPageVisit(),
                 max_results: int = 5,
                 add_wikipedia_results: bool = False):
        self.add_wikipedia_results = add_wikipedia_results
        self.search_tool = DuckDuckGoSearchTool(max_results=max_results)
        self.visit_tool = visit_tool
        self.add_wikipedia_results = add_wikipedia_results
        self.wikipedia_tool = WikipediaRetriever()

    def __call__(self, query: str, **kwargs) -> list[PageResult]:
        search_results = self.search_tool(query)
        if self.add_wikipedia_results:
            search_results += self.search_tool(query+" Wikipedia ")
        title_links = re.findall(r"\[[^)]+\]\(https://[^)]+\)", search_results)
        #remove duplicates
        title_links = list(set(title_links))
        out = []
        #added_links = []
        for result in title_links:
            link = re.findall(r"\((https://[^)]+)\)", result)[0]
            try:
                content = self.visit_tool(link)
            except Exception as e:
                continue
            out.append(PageResult(url=link, source=result, content=content))
            #added_links.append(link.lower())

        # if self.add_wikipedia_results:
        #     out.extend(self.wikipedia_results(query, added_links))
        return out

    # def wikipedia_results(self, query: str, added_links: list[str]) -> list[PageResult]:
    #     wikipedia_results = self.wikipedia_tool.invoke(query)
    #     out = []
    #     for result in wikipedia_results:
    #         if result.metadata["source"].lower() in added_links:
    #             continue
    #         out.append(PageResult(url=result.metadata["source"],
    #                               source=f"[{result.metadata["title"]}]({result.metadata["source"]})",
    #                               content=result.page_content))
    #     return out


class VerificationError(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def load_as_txt(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1]
    if ext in [".txt", ".py", ".md", ".json"]:
        with open(filepath, "r") as f:
            out = f.read()
    elif ext == ".csv":
        out = pd.read_csv(filepath).to_string()
    elif ext == ".xlsx":
        out = pd.read_excel(filepath).to_string()
    else:
        raise ValueError(f"File type {ext} not supported")
    return out
