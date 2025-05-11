from importlib import resources

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool, CodeAgent

from gaia_multiagent import prompts
from gaia_multiagent.cfg import RetrieverCfg, SearchAssistantCfg
from gaia_multiagent.engines import GeminiEngine
from gaia_multiagent.tools.youtube import YouTubeQA
from gaia_multiagent.utils import PlaywrightPageVisit, InternetSearch


class WebResultsRAG(Tool):
    name = "WebSearchRAG"
    description = ("This tool searches on the internet and returns the most relevant passages to the query found in the"
                   " sources. The passages are grouped together by source.")

    inputs = {
        "query": {
            "type": "string",
            "description": "The query to search for. It expects a query in natural language.",
        }
    }

    output_type = "string"

    def __init__(self,
                 websearch_engine: InternetSearch,
                 embeddings: HuggingFaceEmbeddings,
                 cfg: RetrieverCfg = RetrieverCfg()):
        super().__init__()
        self.visit_tool = PlaywrightPageVisit()
        self.embeddings = embeddings
        self.websearch_engine = websearch_engine
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size,
                                                       chunk_overlap=cfg.chunk_overlap,
                                                       separators=cfg.separators)
        self.cfg = cfg

    def get_search_documents(self, query: str) -> list[Document]:
        search_results = self.websearch_engine(query)
        documents = []
        for result in search_results:
            splits = self.splitter.split_text(result.content)
            for s in splits:
                documents.append(Document(page_content=s, metadata={"source": result.source}))
        return documents

    def get_results_vectorstore(self, query: str) -> FAISS:
        documents = self.get_search_documents(query)
        vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        return vector_store

    def forward(self, query: str) -> str:
        vectorstore = self.get_results_vectorstore(query=query)
        results = vectorstore.similarity_search(query=query, k=self.cfg.k)
        sources_data = {}
        for r in results:
            source = r.metadata['source']
            if source not in sources_data:
                sources_data[source] = [r.page_content]
            else:
                sources_data[source].append(r.page_content)
        context = ""
        for source, data in sources_data.items():
            context += f"Source: {source}\n"
            for i, d in enumerate(data):
                context += f"Passage {i}: {d}\n"
            context += "\n"
        return context


class WebSearch(Tool):
    name = "WebSearch"
    description = "This tool finds relevant information about a task on the internet."

    inputs = {
        "query": {
            "type": "string",
            "description": "A natural language query describing what has to be looked up on the internet. "
                           "Be specific about what has to be looked up, provide the complete description of what you "
                           "need.",
        }
    }

    output_type = "string"

    def __init__(self,
                 websearch_engine: InternetSearch,
                 engine: GeminiEngine,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 cfg: RetrieverCfg = RetrieverCfg()):
        super().__init__()
        self.websearch_engine = websearch_engine
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.cfg = cfg
        self.engine = engine

    def forward(self, query: str) -> str:
        web_rag_tool = WebResultsRAG(websearch_engine=self.websearch_engine, embeddings=self.embeddings, cfg=self.cfg)
        agent = CodeAgent(model=self.engine, max_steps=3, tools=[web_rag_tool], verbosity_level=0)
        refined_task = (f"Provide information about the following task: '{query}'. "
                        f"Provide a small summary of what you found."
                        f" Always cite sources urls from which you got each pieace of information.")
        return agent.run(refined_task)


class WebPageRetriever(Tool):
    name = "WebPageRetriever"
    description = ("This tool visits a webpage and returns the information relevant to the query."
                   "If the page can't be reached or the information is incomplete, try with another page. "
                   "Use this tool when you are reasonably confident that the information is in the webpage after having "
                   "used the WebSearch tool.")
    inputs = {
        "task": {
            "type": "string",
            "description": "A natural language query describing what has to be looked up in the page. "
                           "Be specific about what has to be searched.",
        },
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }

    output_type = "string"

    def __init__(self,
                 engine: GeminiEngine,
                 system_prompt: str = resources.read_text(prompts, "page_retriever.txt"),
                 ):
        super().__init__()
        self.engine = engine
        self.visit_tool = PlaywrightPageVisit()
        self.system_prompt = system_prompt

    def forward(self, task: str, url: str) -> str:
        page_content = self.visit_tool(url)
        context = f"<page>{page_content}</page>\n"
        request = f"Find and summarize information in the Webpage related to: '{task}'."
        messages = [{"role": "system", "content": [{"text": self.system_prompt}]},
                    {"role": "user", "content": [{"text": context + request}]}]
        out = self.engine(messages)
        return out.content


# PROBABLY TOO MUCH OVERHEAD

class WebSearchAssistant(Tool):
    name = "WebSearchAssistant"
    description = ("This tool ask an expert assistant to find textual information on the internet. It will return the "
                   "relevant information it has found about an assignment. Do not use "
                   "Provide an assignment including a detailed description of the information that you need. The "
                   "assignment must be specific and not too broadly scoped. It will return a comprehensive answer with "
                   "the relevant information it has found together with the way it has found it.")
    inputs = {
        "assignment": {
            "type": "string",
            "description": "A natural language description of the information you need and its use.",
        },

    }

    task_prompt = ("Find the relevant information regarding the assignment. Provide a comprehensive summary "
                   "of all the information needed to complete the assignment together with explanation of what you have "
                   "done to retrieve it. Don't list the individual tool calls, just summarize the procedure."
                   "\nAssignment: {assignment}")

    output_type = "string"

    def __init__(self,
                 engine: GeminiEngine,
                 search_engine: InternetSearch,
                 cfg: SearchAssistantCfg = SearchAssistantCfg(),
                 download_folder: str = 'tmp_files'):
        super().__init__()
        self.engine = engine
        self.search_engine = search_engine
        self.cfg = cfg
        self.web_search_tool = WebSearch(engine=engine,
                                         embedding_model=self.cfg.embedding_model,
                                         websearch_engine=search_engine,
                                         cfg=self.cfg.retriever_cfg)
        self.web_page_tool = WebPageRetriever(engine=engine)
        self.youtube_tool = YouTubeQA(model_id=engine.model_id, output_dir=download_folder)
        self.agent = CodeAgent(model=engine,
                               tools=[self.web_search_tool, self.web_page_tool, self.youtube_tool],
                               # with web_rag_tools it works
                               planning_interval=self.cfg.planning_interval,
                               verbosity_level=self.cfg.verbosity_level,
                               max_steps=self.cfg.max_steps)
        self.agent.prompt_templates["planning"]["initial_plan"] = resources.read_text(prompts, "initial_planning.txt")

    def forward(self, assignment: str) -> str:
        prompt = self.task_prompt.format(assignment=assignment)
        output = self.agent.run(prompt)
        return output
