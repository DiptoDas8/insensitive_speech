# -*- coding: utf-8 -*-

import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = Ollama(model="deepseek-r1", base_url="http://127.0.0.1:11434")

embed_model = OllamaEmbeddings(
    model="deepseek-r1",
    base_url='http://127.0.0.1:11434',
)

# text = """
#     In the lush canopy of a tropical rainforest, two mischievous monkeys, Coco and Mango, swung from branch to branch, their playful antics echoing through the trees. They were inseparable companions, sharing everything from juicy fruits to secret hideouts high above the forest floor. One day, while exploring a new part of the forest, Coco stumbled upon a beautiful orchid hidden among the foliage. Entranced by its delicate petals, Coco plucked it and presented it to Mango with a wide grin. Overwhelmed by Coco's gesture of friendship, Mango hugged Coco tightly, cherishing the bond they shared. From that day on, Coco and Mango ventured through the forest together, their friendship growing stronger with each passing adventure. As they watched the sun dip below the horizon, casting a golden glow over the treetops, they knew that no matter what challenges lay ahead, they would always have each other, and their hearts brimmed with joy.
#     """
text = """
    মাদারীপুরের কালকিনিতে বন্ধ করে দেওয়া হলো শত বছরের ঐতিহ্যবাহী কুন্ডুবাড়ির মেলা। প্রায় আড়াই শ বছরের পুরোনো এ মেলা প্রতি বছর কালিপূজায় সপ্তাহব্যাপী অনুষ্ঠিত হলেও এ বছর তা আর হচ্ছে না। উপজেলা প্রশাসনের দাবি, স্থানীয় জনগণ ও আলেম সমাজ এ মেলা আয়োজনে আপত্তি জানায়। এ কারণে প্রশাসন মেলা আয়োজনে অনুমতি দেয়নি। বিষয়টি নিশ্চিত করে কালকিনি উপজেলা নির্বাহী কর্মকর্তা উত্তম কুমার দাশ বলেন, কালকিনি উপজেলার ভুরঘাটার কুন্ডুবাড়ির ঐতিহ্যবাহী কালিপূজাকে কেন্দ্র করে মূলত সপ্তাহব্যাপী মেলা অনুষ্ঠিত হয়ে আসছিল। এ মেলা এবারও অনুষ্ঠানের জন্য কালকিনি পৌরসভা থেকে আকবর হোসেন নামের এক ব্যক্তিকে ইজারা দেওয়া হয়েছিল। কিন্তু স্থানীয় আলেম সমাজ ও ছাত্র প্রতিনিধিরা মেলায় আইন‑শৃঙ্খলার অবনতিসহ বিভিন্ন সমস্যা উল্লেখ করে মেলা বন্ধ করার জন্য লিখিত অভিযোগ দেয়। তাদের অভিযোগের পরিপ্রেক্ষিতে কালকিনি পৌরসভা ও উপজেলা প্রশাসন এবার মেলা আয়োজন বন্ধ করার সিদ্ধান্ত নেয়। তবে মেলা বন্ধ থাকলেও মন্দিরটিতে কালিপূজা যথারীতি অনুষ্ঠিত হবে উল্লেখ করে উপজেলা নির্বাহী কর্মকর্তা বলেন, ধর্মীয় রীতি অনুযায়ী কুন্ডুবাড়ির কালিপূজা যথারীতি অনুষ্ঠিত হবে। তবে পূজাকে কেন্দ্র করে হওয়া শুধু মেলা বন্ধ থাকবে। এ বিষয়ে মেলা আয়োজনের ইজারাদার আকবর হোসেন বলেন, প্রায় আড়াই শ বছর ধরে কুন্ডুবাড়ির মেলা হয়ে আসছে। কেউ কোনোদিন প্রশ্ন তোলেনি। কিন্তু এবার প্রশাসনের কাছে আলেম সমাজ আইন‑শৃঙ্খলা পরিস্থিতির অবনতি হওয়ার আশঙ্কা করে মেলা না করার জন্য লিখিত অভিযোগ দিয়েছে। কিন্তু তারা আমার কাছে এসে বলেছে, হিন্দুদের মেলায় মুসলমানরা যাবে না। তাই এখানে মেলা করা যাবে না। আমাকে তারা মেলা না করার জন্য নিষেধ করেছে। মেলা না হওয়ায় আর্থিক লোকসানের মুখে পড়বেন জানিয়ে ইজারাদার আকবর হোসেন আরও বলেন, আমি পৌরসভা থেকে মেলার ইজারা নিয়ে কয়েক শ দোকানির কাছে জায়গা বরাদ্দ দিয়েছি। এসব দোকানিরা ঋণ করে লাখ লাখ টাকার মালামাল মেলায় বিক্রির জন্য প্রস্তুতি গ্রহণ করেছে। এখন এই মেলা না হলে এসব দোকানি ঋণের জালে জড়িয়ে পড়বে। তাছাড়া আমারও আর্থিক অনেক ক্ষতি হবে। আমি এ বিষয়ে দ্রুতই পৌর কর্তৃপক্ষের কাছে ক্ষতিপূরণ চেয়ে আইনি পদক্ষেপ গ্রহণ করবো। মেলা না হওয়ার ব্যাপারে জানতে চাইলে মন্দির কমিটির উপদেষ্টা বাসু দেব কুণ্ডু বলেন, দিপাবলী ও শ্রী শ্রী কালিপূজা উপলক্ষে আমাদের পূর্বপুরুষরা এই মেলা আয়োজন করে আসছেন। এবার মেলা কারা বন্ধ করে দিয়েছে–এটি আপনারা খুঁজে বের করুন। বাজারে কারা ব্যানার টাঙিয়েছে মেলা না করার জন্য–সেটা খুঁজে বের করার দায়িত্ব আপনাদের। আর বেশি কিছু বলে আমরা বিপদে পড়তে চাই না।
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

vector_store = FAISS.from_texts(chunks, embed_model)

retriever = vector_store.as_retriever()

chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# query = "Tell me name of monkeys and where do they live"
prompt = 'As a supportive and nurturing content moderator like a teacher, evaluate whether this text could be perceived as hate speech, hurtful, or culturally insensitive. Consider if it marginalizes, reinforces stereotypes, or excludes any group. If so, provide constructive feedback by identifying concerns, explaining why they may be problematic, and suggesting more inclusive alternatives. Answer briefly and translate that in the Bengali language before responding.'
question = 'আলহামদুলিল্লাহ প্রায় ৩০০ বছরের পূর্ব থেকে সংগঠিত কুন্ডু বাড়ির মেলা আর কোনো দিন সংগঠিত হবেনা। ইনশাল্লাহ।'
query = f'{prompt}\n\n{question}'
response = retrieval_chain.invoke({"input": query})
print(response['answer'])

with open('rag_response_test.txt', 'w', encoding='utf-8') as fout:
    fout.write(response['answer'])