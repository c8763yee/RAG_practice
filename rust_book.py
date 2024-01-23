import bs4
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import OpenAICallbackHandler
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = hub.pull("rlm/rag-prompt")
load_dotenv(override=True)


URL = 'https://doc.rust-lang.org/book'


def get_rust_book_chapter_URL() -> list[str]:
    loader = requests.get(URL)
    soup = bs4.BeautifulSoup(loader.content, 'html.parser')
    chapter_list = soup.find("ol", attrs={"class": "chapter"}).find_all("a")
    return [f"{URL}/{chapter.attrs['href']}" for chapter in chapter_list]


def load_rust_book_chapter_content(chapter_urls: list[str]) -> list[str]:
    loader = WebBaseLoader(
        web_path=chapter_urls,
        bs_kwargs={"parse_only": bs4.SoupStrainer(
            "main")}
    )
    return loader.load()


def split_rust_book_chapter_content(chapter_contents: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )
    return splitter.split_documents(chapter_contents)


def main():
    chapter_urls = get_rust_book_chapter_URL()
    chapter_contents = load_rust_book_chapter_content(chapter_urls)
    # with open("rust_book.txt", "w") as f:
    #     f.write("\n".join(map(lambda Document: Document.page_content, chapter_contents)))

    splited_content = split_rust_book_chapter_content(chapter_contents)
    # for chapter_content in chapter_contents:
    #     print(chapter_content.page_content)
    vectorstore = Chroma.from_documents(
        documents=splited_content, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 20})

    cb = OpenAICallbackHandler()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106",
                     temperature=0.3, callbacks=[cb])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm | StrOutputParser()
    )
    question = input("question: ")

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
