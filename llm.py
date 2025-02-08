from ollama import chat
from ollama import ChatResponse

prompt = 'As a supportive and nurturing content moderator like a teacher, evaluate whether this text could be perceived as hate speech, hurtful, or culturally insensitive. Consider if it marginalizes, reinforces stereotypes, or excludes any group. If so, provide constructive feedback by identifying concerns, explaining why they may be problematic, and suggesting more inclusive alternatives. Answer briefly and translate that in the Bengali language before responding.'
question = 'আলহামদুলিল্লাহ প্রায় ৩০০ বছরের পূর্ব থেকে সংগঠিত কুন্ডু বাড়ির মেলা আর কোনো দিন সংগঠিত হবেনা। ইনশাল্লাহ।'

response: ChatResponse = chat(model='deepseek-r1', messages=[
    {
        'role': 'system',
        'content': prompt,
    },
    {
        'role': 'user',
        'content': question
    }
])

print(response['message']['content'])
response_text = response.message.content

with open('response_test.txt', 'w', encoding='utf-8') as fout:
    fout.write(response_text)