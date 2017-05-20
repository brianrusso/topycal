# topycal

Do you want to apply topic modelling to your data but don't know what a text vectorizer or a topic distribution matrix is?
Do you simply want to throw a list of simple documents/dictionaries against one and enrich your data with the topic clusters?

Topycal is for you and a simple pip3 install topycal away!

It is a convenient wrapper against the excellent NMF/LDA implementations in sklearn. If you want a more sophisticated interaction with topic modelling; I suggest you checkout the excellent gensim package or the underlying sklearn functions used here.

1. Start with an array of simple documents. These are not very interesting of course.. just an example.
```python
mydocs = [
  # Opera topic
  {"doc_id":1,"text":"My mom drove me to the opera fifteen minutes late on Tuesday."},
  {"doc_id":11,"text":"My bandaid wasn't sticky any more so it fell off on the way to the opera."},
  {"doc_id":5,"text":"The Lexus' door slammed down on my hand and I screamed like a little baby."},
  {"doc_id":8,"text":"I was so thirsty at the opera, I couldn't wait to get a glass of wine."},
  {"doc_id":9,"text":"I found a gold coin by a Lexus in the parking lot after the opera performance today."},
  {"doc_id":15,"text":"Your mom drives a Lexus, and she is so nice she gave me a ride home today from the opera."},
  # School topic
  {"doc_id":2,"text":"The girl wore her hair in two braids, tied with two blue bows. She lost a bow in the school playground."},
  {"doc_id":14,"text":"I was so scared to go to the Opera but my dad said he would sit with me so we went last night."},
  {"doc_id":6,"text":"My shoes are blue with yellow stripes and green stars on the front."},
  {"doc_id":16,"text":"I fell in the mud when I was walking home from school today."},  
  {"doc_id":17,"text":"This kids at school were so hungry they couldn't stop eating the tasty school cafeteria food."},
  # Zoo topic  
  {"doc_id":3,"text":"The giraffe was so hungry he ran across the kitchen floor without even looking for humans."},
  {"doc_id":4,"text":"The hippo got stuck in the zoo's carousel and he couldn't talk anymore."},
  {"doc_id":12,"text":"The hippo had a sore throat so I gave him my bottle of water from the zoo's store and told him to keep it."},
  {"doc_id":13,"text":"The zebra was white and brown and looked very old compared to the giraffe. The school kids were curious about his health."},
  {"doc_id":7,"text":"The giraffe's cage was bent and broken and looked like a hippo had knocked it over on purpose."},
  {"doc_id":10,"text":"The chocolate chip cookies smelled so good that the hippo ate one without asking."}]  
  
```

2. Instantiate a model (you could also do TopycalLDA) + pass in the docs. Since this is a tiny corpus, we select 3 topics and 5 topic words.
```python
model = TopycalNMF(mydocs, num_topics=3, num_topic_words=5, content_key='text')
```

3. Initialize the topic model
```python
model.initialize()
```

4. Now you can easily get topics assigned for each document
```python
model[3]
=> (0, 0.4285221719166587)
```

5. But what is topic 0? We can look at the topic words to get an idea.
```python
model.topics
=>
{0: ['opera', 'today', 'lexus', 'fell', 'couldn'],
 1: ['hippo', 'zoo', 'couldn', 'gave', 'like'],
 2: ['school', 'blue', 'kids', 'giraffe', 'hungry']}
```

6. Okay, looks like it's about rich people at the opera. Let's give the topics human names
```python
model.set_topic_names(['Night Life', 'Wild Animals', 'Schooltime'])
```

7. Now when we get the topic, it'll give us a human name
```python
model[3]
=> ('Night Life', 0.4285221719166587)
```
8. Or you can still force a topic id if you prefer.
```python
model.get_topic_for_doc(3, force_topicid=True)
=> (0, 0.4285221719166587)
```

9. If you set a threshold you can get the thresholded' topic vectors for a given document. If you set it to 0 you get all the vectors.
```python
model.get_topic_for_doc(3, threshold=0.05, force_topicid=True)
=> [(0, 0.4285221719166587), (1, 0.091668549635229787)]

model.get_topic_for_doc(3, threshold=0)
=>
[('Night Life', 0.4285221719166587),
 ('Wild Animals', 0.091668549635229787),
 ('Schooltime', 0.0022845521247407997)]
```

10. Finally, if you set a topic_key, you will change the sequence behaviour to output the original document with the topic in whatever key you define.
```python
model.set_topic_key("topics")
model[3]
=>
{'doc_id': 8,
 'text': "I was so thirsty at the opera, I couldn't wait to get a glass of wine.",
 'topics': ('Night Life', 0.4285221719166587)}
# Disable by setting topic_key to None
model.set_topic_key(None)
model[3]
=> ('Night Life', 0.4285221719166587)
```
