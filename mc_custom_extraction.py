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
import random

from alphabet_detector import AlphabetDetector
import six

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "The input data directory containing the raw dump file.")
flags.DEFINE_string("output_dir", None, "The output directory to where .txt file will be written into.")
flags.DEFINE_string("language", None, "The language of interest.")
flags.DEFINE_integer("char_count_lower_bound", None, "The upper bound character count.")
flags.DEFINE_integer("char_count_upper_bound", None, "The lower bound character count.")
flags.DEFINE_integer("output_num_examples", None, "The number of examples in final output text file.")



lang_group_mapper = {
  "en": "LATIN",
  "es": "LATIN",
  "de": "LATIN",
  "zh": "CJK",
  "ar": "ARABIC",
  "ur": "ARABIC"
}

ad = AlphabetDetector()



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
  if ((line.startswith("<")) or (line.startswith("http")) or (line.startswith("!")) or (line.startswith("www")) or (line in ["\n"])) or (not (lang_group_mapper[language] in ad.detect_alphabet(uni(line)))):
	return False
  else:
	return True



def main():

  lower_bound = FLAGS.char_count_lower_bound
  upper_bound = FLAGS.char_count_upper_bound

  fr = open(os.path.join(FLAGS.data_dir, "{}".format(FLAGS.language)), "r")
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