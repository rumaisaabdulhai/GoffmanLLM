# def top_topics_prompt(audience, topics_list, num_topics):
#   return f"You are at a dinner with {audience}. Here is a list of topics you could discuss with them: {topics_list}. \
#   Please select the top {num_topics} topics you would most likely discuss and only report the names of the topics. Example of your answer would look like: Topic 1, Topic 2, Topic 3"

def top_topics_prompt(audience, topics_list, num_topics):
  return f"You are at a dinner with {audience}. Here is a list of topics you could discuss with them: {topics_list}. \
  Please select the top {num_topics} topics you would most likely discuss and only report the topic names. Example response: Topic 1, Topic 2, Topic 3"


def yes_or_no_topic_prompt(audience, topic):
  return f"You are at a dinner with {audience}. Would you discuss this topic with them: {topic}? Report 1 if you would talk about it or 0 if you would not talk about it. Please just output the number."

def formal_prompt(audience):
  return f"You are at a dinner with {audience}. Would you speak casually or more formally with them? Please report 1 for casually and 0 for formally."

def setting_prompt(audience):
  return f"You are at a dinner with {audience}. Where would the dinner most likely happen: 1) at someone's home, 2) at a cafe, 3) at a bar, or 4) at a fancy restaraunt? Please report only the number."
