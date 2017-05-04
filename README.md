# topycal
Python Topic Modelling Framework

Topycal is a topic modelling framework that (currently) exposes the SKLearn NMF and LDA models in an easy to use way.

del = Topycal()

1. Start with an array of simple documents. These are not very interesting of course.. just an example.

mydocs = [
  {"doc_id":1,"text":"My mom drove me to school fifteen minutes late on Tuesday."},
  {"doc_id":2,"text":"The girl wore her hair in two braids, tied with two blue bows."},
  {"doc_id":3,"text":"The mouse was so hungry he ran across the kitchen floor without even looking for humans."},
  {"doc_id":4,"text":"The tape got stuck on my lips so I couldn't talk anymore."},
  {"doc_id":5,"text":"The door slammed down on my hand and I screamed like a little baby."},
  {"doc_id":6,"text":"My shoes are blue with yellow stripes and green stars on the front."},
  {"doc_id":7,"text":"The mailbox was bent and broken and looked like someone had knocked it over on purpose."},
  {"doc_id":8,"text":"I was so thirsty I couldn't wait to get a drink of water."},
  {"doc_id":9,"text":"I found a gold coin on the playground after school today."},
  {"doc_id":10,"text":"The chocolate chip cookies smelled so good that I ate one without asking."},
  {"doc_id":11,"text":"My bandaid wasn't sticky any more so it fell off on the way to school."},
  {"doc_id":12,"text":"He had a sore throat so I gave him my bottle of water and told him to keep it."},
  {"doc_id":13,"text":"The church was white and brown and looked very old."},
  {"doc_id":14,"text":"I was so scared to go to a monster movie but my dad said he would sit with me so we went last night."},
  {"doc_id":15,"text":"Your mom is so nice she gave me a ride home today."},
  {"doc_id":16,"text":"I fell in the mud when I was walking home from school today."},
  {"doc_id":17,"text":"This dinner is so delicious I can't stop eating."}]

mydocs = [{'title': 'Mr Smith watches TV', 'text': 'A heartwarming tale about Bob Smith watching television on a Sunday afternoon'},
          {'title': 'Joe plays tennis', 'text': 'A riveting story of a man who learns to play tennis and satisfy his dreams'}]

2. Instantiate + pass in the array
topicmodel = Topycal(mydocs, 'text')

3. Generate the topic model, LDA in this case.
topicmodel.model_with_lda()

4. Now you can just easily get topics assigned for each document
topicmodel[5]

{'doc_id': 6,
 'text': 'My shoes are blue with yellow stripes and green stars on the front.',
 'topics': [9]}
