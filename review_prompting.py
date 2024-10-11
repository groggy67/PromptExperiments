# Testing my local ollama-haystack and local llama
import logging
import json

from schema import review_schema

from haystack import  Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

logging.basicConfig(filename='review.pipeline.log', level=logging.INFO)
logging.info('Start')

text = "After so many positive reviews of this hotel there's not much left to say. With that said, here's my 2 cents worth. The Homewood Suites is very comfortable within walking distance to attractions and restaurants. There's a friendly, helpful front desk staff. The breakfast buffet is plentiful and is kept well stocked. If your stay is on a weekday the hotel offers a dinner buffet and complimentary wine and beer. Clean, comfortable rooms and very affordable rates for D.C. I'd stay here again in a minute."

query = """
Determine if the text contains a sentiment about the following,
hotel location,
hotel room.
Format your response as a JSON object with the following keys,
'location',
'location_sentiment',
'room',
'room_sentiment',
'cleaning',
'cleaning_sentiment',
'food',
'food_sentiment'
'staff_service',
'staff_service_sentiment'
'cost',
'cost_sentiment',
'facilities',
'facilities_sentiment',
'bed',
'bed_sentiment',
'wifi',
'wifi_sentiment',
'noise',
'noise_sentiment',
'anger_sentiment'
Any sentiment should be either 'positive', 'neutral' or 'negative'.
The location key value will be the comment about the location of the hotel and nothing else.
The location_sentiment key value will be the sentiment of the comment about the location of the hotel and nothing else.
The room key value will be the comment about the room and nothing else.
The room_sentiment key value will be the sentiment of the comment about the room and nothing else.
The cleaning key value will be the comment about any dirt, dirty, soiled, filth, unitidiness, unclean, lack or hygiene, stains in the hotel and nothing else.
The cleaning_sentiment key value will be the sentiment about any dirt, dirty, soiled, filth, unitidiness, unclean, lack or hygiene, stains  in the hotel and nothing else.
The food key value will be the comment about the food and nothing else.
The food_sentiment key value will be the sentiment of the comment about the food and nothing else.
The staff_service key value will be the comment about the any staff members and nothing else.
The staff_service_sentiment key value will be the sentiment of the comment about the staff and nothing else.
The cost key value will be the comment about the price of room or cost of hotel but not about any prices of things outside the hotel and nothing else.
The cost_sentiment key value will be the sentiment of the price of room and or cost of hotel and nothing else.
The facilities key value will be the comment about the hotel facilities such as shuttle buses, parking area, gym, swimming pool, public area.
The facilities_sentiment key value will be the sentiment of the hotel facilities such as shuttle buses, parking area, gym, swimming pool, public area.
The bed key value will be the sentence which must contain the word the words  'bed' or 'beds' or 'duvet' or 'sheet' or 'sheets' or 'mattress' or 'blankets' or 'pillows'  and nothing else otherwise the bed key value is 'unknown'.
The bed_sentiment key value will be the sentiment of sentence which must contain the word  'bed' or 'beds' or 'pillow menu' or 'duvet' or 'sheet' or 'sheets' or 'mattress' or 'blankets' or 'pillows'  and nothing else. otherwise the bed_sentiment  key value is 'neutral'.
The wifi key value will be the comment about wifi or internet used in hotel and nothing else.
The wifi_sentiment key value will be the sentiment about wifi or internet used in hotel and nothing else.
The noise key value will be the sentence containing the words 'noise' or 'noisy'  and nothing else.
The noise_sentiment key value will be 'negative' if the sentence contains the words 'noise' or 'noisy'  otherwise it will be 'neutral'.
The anger_sentiment key value will be the true if the tone  or sentiment of the comment is angry and false if not.
If the information isn't present, use 'unknown' as the value. 
If the sentiment is isn't present use 'neutral' as the value.
Just return the JSON object as the answer. 
"""

template = """  
Give the following information, answer the question from the information present in this text: {{text}}.
Ignore your on knwoledge.
Question={{query}}?
"""


pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template=template ))
pipe.add_component("llm", OllamaGenerator(model="llama3.2:latest", url="http://localhost:11434"))
pipe.connect("prompt_builder", "llm")


response = pipe.run({"prompt_builder": {"query": query, "text": text }})
print(json.dumps(response["llm"]["replies"], indent=4))
