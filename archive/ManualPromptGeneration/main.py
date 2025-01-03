import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import words
import nltk

# NLTK stop words
#nltk.download('stopwords')
#nltk.download('words')

# Filtering the text
def process_text(input_text):
    # Step 1: Remove "[Music]"
    text = input_text.replace("[Music]", "")

    # Step 2: Remove numbers
    text = re.sub(r'\d+', '', text)

    # Step 3: Remove character strings, that are no words
    english_words = set(words.words())
    words_in_text = text.split()
    valid_words = [word for word in words_in_text if word.lower() in english_words]

    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in valid_words if word.lower() not in stop_words]

    # Merge to text
    filtered_text = ' '.join(filtered_words)
    return filtered_text


# Find the most common words
def get_top_words(filtered_text, top_n=10):
    words = filtered_text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(top_n)
    return most_common_words


# Example
input_text = """
0:10
oh you better run and hide baby I know
0:14
where you were last night so I'm
0:17
lighting up the sky setting fire to your
0:21
Paradise you ain't got no you ain't got
0:24
no you ain't got no Al you ain't got no
0:29
you ain't got no oh you ain't got no
0:32
alibi you better run and hide baby I
0:36
know where you were last night so I'm
0:39
lighing up the sky setting fire to your
0:43
Paradise you better running High baby I
0:47
know where you were last night so I'm
0:50
lining up the sky setting fire to your
0:54
Paradise you ain't going to know you
0:56
ain't going to know you ain't going to
0:58
know
1:00
you ain't got no you ain't got no you
1:03
ain't got
1:05
no got to say I'm sorry and I made me
1:09
Mama never tell you how to treat a lady
1:11
by this time tomorrow you going to
1:13
really hate me bet you wish you never
1:15
play me you took her in I back you got
1:19
no respect why did you now I'm burn this
1:22
house down to the leg I'mma to the next
1:25
I'mma get my sweet revenge tell me how
1:29
you li
1:30
so
1:32
weily You' been messing with her don't
1:36
mess with me tell me how you li so
1:43
Reas you've been messing with her don't
1:47
mess with
1:49
me you better run and hide baby I know
1:53
where you were last night so I'm
1:56
lighting up the sky setting fire to your
2:00
Paradise you better run and High baby I
2:04
know where you were last night so I'm
2:07
lying up the sky set fire to your
2:11
Paradise you ain't got you ain't got no
2:15
you ain't got you
2:20
got
2:23
got got you got
2:28
got you got
2:38
you got
2:42
nobody
2:46
oh
2:51
you
2:53
you got
2:56
[Music]
3:00
oh
"""

filtered_text = process_text(input_text)
print("Cleaned text:", filtered_text)

top_words = get_top_words(filtered_text)
print("The 10 most common words:")
for word, count in top_words:
    print(f"{word}: {count}")
common_words = [word for word, count in top_words]
prompt = ' '.join(common_words)
print(prompt)