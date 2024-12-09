# %%
import pandas as pd
import random
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from prompts import *
from constants import *

seed = 12
random.seed(seed)

# %%
MODEL = "gpt-4o-mini"
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# %%
def backstory_chat(persona, prompt):
  memory = [{"role": "system", "content": f"You are the person in the following backstory {persona}"},
            {"role": "user", "content": f"{prompt}"}]
  response = client.chat.completions.create(
      model=MODEL,
      messages=memory,
  )
  answer = response.choices[0].message.content
  return answer

def get_personas():
    with open('anthology_personas.pkl', 'rb') as f:
        backstories = pickle.load(f)
    return backstories

# %%
def experiment1(personas, num_trials=3, k=3):
    """
    The goal of this experiment is to see which general topics are included in the top k topics for a particular audience over all personas.
    We want to see which topics are prioritized.
    """
    answers = []
    for trial in tqdm(range(num_trials)):
        trial_answers = []
        for persona in tqdm(personas, leave=False):
            persona_answers = []
            for audience in tqdm(audiences, leave=False):
                answer = backstory_chat(persona, top_topics_prompt(audience, general_topics, k))
                persona_answers.append(answer)
            trial_answers.append(persona_answers)
        answers.append(trial_answers)

    
    # table of audience vs. general topics: was this general topic mentioned in the top 3 for the audience?
    table = np.zeros((len(audiences), len(general_topics)))
    general_topics_lower = [x.lower() for x in general_topics]

    for trial in tqdm(range(num_trials)):
        for persona_idx in range(len(personas)):
            for audience_idx in range(len(audiences)):
                print(answers[trial][persona_idx][audience_idx])
                audience_answers = answers[trial][persona_idx][audience_idx]
                top_k_topics = audience_answers.split(", ")
                top_k_topics = [answer.strip(". ").lower() for answer in top_k_topics]
                for topic in top_k_topics:
                    topic_idx = general_topics_lower.index(topic)
                    table[audience_idx][topic_idx] += 1

    df = pd.DataFrame(table, index=audiences, columns=general_topics)
    df /= num_trials
    df.to_pickle("experiment1/table.pkl")

# %%
experiment1(personas)

# %%
df = pd.read_pickle("experiment1/table.pkl")
df = df.round(2)
html_table = df.to_html(index=False)
with open("experiment1/table.html", "w") as file:
    file.write(html_table)

# %%
def topic_proportions_fig(df, file_name):
    # Topic Proportions over all Audiences
    post_types = df.columns.tolist()
    post_frequencies = [df[col].mean() for col in post_types]
    filtered_post_types = [post_types[i] for i in range(len(post_types)) if post_frequencies[i] > 0]
    filtered_post_frequencies = [post_frequencies[i] for i in range(len(post_frequencies)) if post_frequencies[i] > 0]
    plt.pie(filtered_post_frequencies, labels=filtered_post_types, autopct='%1.1f%%')
    _ = plt.title('Topic Proportions Over All Audiences')
    plt.savefig(file_name)

# %%
topic_proportions_fig(df, 'experiment1/topic_proportions.png')

# %%
def experiment2(personas, num_trials=3):
    """
    """
    responses = []
    for trial in tqdm(range(num_trials)):
        response_per_persona = []
        for persona in tqdm(personas, leave=False):
            response_per_audience = []
            for audience in tqdm(audiences, leave=False):
                response_per_topic = []
                for topic in tqdm(general_topics, leave=False):
                    response = backstory_chat(persona, yes_or_no_topic_prompt(audience, topic))
                    response_per_topic.append(response)
                response_per_audience.append(response_per_topic)
            response_per_persona.append(response_per_audience)
        responses.append(response_per_persona)

    table2 = np.zeros((len(audiences), len(general_topics)))

    for trial in tqdm(range(num_trials)):
        for i in tqdm(range(len(personas)), leave=False):
            for j in tqdm(range(len(audiences)), leave=False):
                for k in tqdm(range(len(general_topics)), leave=False):
                    table2[j][k] += int(responses[trial][i][j][k])

    table2 /= num_trials
    df2 = pd.DataFrame(table2, index=audiences, columns=general_topics)
    df2.to_pickle("experiment2/table.pkl")

# %%
experiment2(personas)

# %%
df2 = pd.read_pickle("experiment2/table.pkl")
df2 = df2.round(2)
html_table = df2.to_html(index=False)
with open("experiment2/table.html", "w") as file:
    file.write(html_table)

# %%
topic_proportions_fig(df2, 'experiment2/topic_proportions.png')
