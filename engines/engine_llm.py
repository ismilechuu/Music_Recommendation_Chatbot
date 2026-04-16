from openai import OpenAI

def call_llm(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"
    )
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def build_prompt(query, context):
    return f"""
        You are a music recommendation assistant.

        User request:
        {query}

        Here are some candidate songs:
        {context}

        Instructions:
        - Recommend 3 songs
        - Explain why each song matches the user's requests
        - Only use songs from the list
        - Do NOT make up songs
        - Answer in a friendly tone

        Answer:
        """

def format_context(df):
    context = ""
    for _, row in df.iterrows():
        context += f"""
        Song: {row['song']}
        Artist: {row['artist']}
        Genre: {row['genre']}
        Mood: {row['emotion']}
        Lyrics: {row['chunks'][:200]}
        ---
        """
    return context

def recommend_songs(query, df_results):
    context = format_context(df_results)
    prompt = build_prompt(query, context)
    answer = call_llm(prompt)
    return answer
