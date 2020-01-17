# Copyright 2020 Superb AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Authors: Channy Hong, Jaeyeon Lee, Jung Kwon Lee.
Description: The monolingual corpora custom extraction script on once-extracted text (Wikipedia) dump.
'''

import tensorflow as tf
import unicodedata as ud
import random
import six
import os

from collections import defaultdict

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("source_file_path", None, "The input data directory containing the raw dump file.")
flags.DEFINE_string("output_dir", None, "The output directory to where .txt file will be written into.")
flags.DEFINE_string("language", None, "The language of interest.")
flags.DEFINE_integer("char_count_lower_bound", None, "The upper bound character count.")
flags.DEFINE_integer("char_count_upper_bound", None, "The lower bound character count.")
flags.DEFINE_integer("output_num_examples", None, "The number of examples in final output text file.")



class AlphabetDetector:
    def __init__(self, no_memory=False):
        self.alphabet_letters = defaultdict(dict)
        self.no_memory = no_memory

    def is_in_alphabet(self, uchr, alphabet):
        if self.no_memory:
            return alphabet in ud.name(uchr)
        try:
            return self.alphabet_letters[alphabet][uchr]
        except KeyError:
            return self.alphabet_letters[alphabet].setdefault(
                uchr, alphabet in ud.name(uchr))

    def only_alphabet_chars(self, unistr, alphabet):
        return all(self.is_in_alphabet(uchr, alphabet)
                   for uchr in unistr if uchr.isalpha())

    def detect_alphabet(self, unistr):
        return set(ud.name(char).split(' ')[0]
                   for char in unistr if char.isalpha())

    def is_greek(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'GREEK') else False

    def is_cyrillic(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'CYRILLIC') else False

    def is_latin(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'LATIN') else False

    def is_arabic(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'ARABIC') else False

    def is_hebrew(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HEBREW') else False

    # NOTE: this only detects Chinese script characters (Hanzi/Kanji/Hanja).
    # it does not detect other CJK script characters like Hangul or Katakana
    def is_cjk(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'CJK') else False

    def is_hangul(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HANGUL') else False

    def is_hiragana(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'HIRAGANA') else False

    def is_katakana(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'KATAKANA') else False

    def is_thai(self, unistr):
        return True if self.only_alphabet_chars(unistr, 'THAI') else False



lang_group_mapper = {
  "en": "LATIN",
  "es": "LATIN",
  "de": "LATIN",
  "zh": "CJK",
  "ar": "ARABIC",
  "ur": "ARABIC"
}



# Converts `text` to Unicode (if it's not already), assuming utf-8 input.
def uni(text):
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def meets_criteria(line, language):
  ad = AlphabetDetector()
  if ((line.startswith("<")) or (line.startswith("http")) or (line.startswith("!")) or (line.startswith("www")) or (line in ["\n"])) or (not (lang_group_mapper[language] in ad.detect_alphabet(uni(line)))):
	return False
  else:
	return True



def main():

  lower_bound = FLAGS.char_count_lower_bound
  upper_bound = FLAGS.char_count_upper_bound

  fr = open(os.path.join(FLAGS.source_file_path, "{}".format(FLAGS.language)), "r")
  candidates = []

  for i, line in enumerate(fr):
    if not meets_criteria(line, FLAGS.language):
      continue
  else:
    if (lower_bound < len(line)) and (len(line) < upper_bound):
      candidates.append(line)

  final_examples = random.sample(candidates, FLAGS.output_num_examples)
  fw = open(os.path.join(FLAGS.output_dir, "{}.txt".format(FLAGS.language)), "w")

  for example in final_examples:
    fw.writelines(example)



##############################
##### The Program Driver #####
##############################

if __name__ == "__main__":
  main()