from openai import OpenAI

def recommend_songs(query, df_results, chat_history=None):
    if chat_history is None:
        chat_history = []

    context = ""
    if df_results is not None:
        for _, row in df_results.iterrows():
            context += f"Song: {row['song']}\nArtist: {row['artist']}\nGenre: {row['genre']}\nMood: {row['emotion']}\nLyrics: {row.get('chunks', '')[:200]}\n---\n"

    system_message = f"""You are a helpful and interactive music recommendation chatbot.
Your goal is to help find the perfect song by engaging in a conversation.

Here are the LATEST candidate songs fetched from the database based on the user's latest query:
{context}

Instructions:
1. Recommend up to 3 songs from the provided candidate list.
2. Explain why they match the user's requests (both current and past context).
3. If the user is providing feedback (e.g. "I want something more upbeat" or "I want it faster"), acknowledge their feedback and explain how the new recommendations meet their new criteria.
4. Keep the original vibe in mind but shift according to the user's latest request.
5. Try to only recommend songs from the provided candidate list.
6. Answer in a friendly tone and always end by asking for their feedback (e.g., "How do you like these? Want something even more upbeat?").
7. If the user asks to "find more" or "show more", recommend more songs from the candidate list (up to 3 at a time).
8. Only recommend songs that clearly match the user's latest request.If none match well, say so.
STRICT RULE:
- Do NOT recommend songs that do not clearly match the user's latest request.
- If a song does not match the requested mood or theme, you MUST exclude it.
- If none of the songs match well, say: "I couldn't find a good match from the current list."
"""

    messages = [{"role": "system", "content": system_message}]

    # append conversation history
    for turn in chat_history:
        messages.append({"role": "user", "content": turn['query']})
        if turn['recommendation']:
            messages.append({"role": "assistant", "content": turn['recommendation']})

    # append current query
    messages.append({"role": "user", "content": query})

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content
