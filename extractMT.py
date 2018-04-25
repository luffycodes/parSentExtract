from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from itertools import product
from six.moves import xrange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf

import utils
import pandas as pd


tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints and meta graph.")

tf.flags.DEFINE_string("extract_dir", "",
                       "Directory containing the aligned articles to do "
                       "parallel sentence extraction.")

tf.flags.DEFINE_string("source_vocab_path", "",
                       "Path to source language vocabulary.")

tf.flags.DEFINE_string("target_vocab_path", "",
                       "Path to target language vocabulary.")

tf.flags.DEFINE_string("source_output_path", "",
                       "Path to the file containing the extracted sentences in "
                       "the source language.")

tf.flags.DEFINE_string("target_output_path", "",
                       "Path to the file containing the extracted sentences in "
                       "the target language.")

tf.flags.DEFINE_string("score_output_path", "",
                       "Path to the file containing the probability scores of "
                       "the extracted sentence pairs.")

tf.flags.DEFINE_string("source_language", "",
                       "Source language suffix used as file extension.")

tf.flags.DEFINE_string("target_language", "",
                       "Target language suffix used as file extension.")

tf.flags.DEFINE_float("decision_threshold", 0.99,
                      "Decision threshold to predict a positive label.")

tf.flags.DEFINE_integer("batch_size", 500,
                        "Batch size to use during evaluation.")

tf.flags.DEFINE_integer("max_seq_length", 100,
                        "Maximum number of tokens per sentence.")

tf.flags.DEFINE_boolean("use_greedy", True,
                        "Use greedy post-treatment to force one-to-one "
                        "alignments.")


FLAGS = tf.flags.FLAGS


def read_articles(source_path, target_path):
    """Read the articles in source and target languages."""
    with open(source_path, mode="r", encoding="utf-8") as source_file,\
         open(target_path, mode="r", encoding="utf-8") as target_file:
            source_sentences = [l for l in source_file]
            target_sentences = [l for l in target_file]
    return source_sentences, target_sentences


def inference(sess, source_path, target_path, source_vocab, target_vocab, probs_op, placeholders, source_final_state_ph):
    """Get the predicted class {0, 1} of given sentence pairs."""

    # Read sentences from articles.
    freeResponseCosineWithCorrectAns = pd.read_csv("Mt_extract.csv")

    for index, row in freeResponseCosineWithCorrectAns.iterrows():
        source_sentences = [row['freeResponse'], row['correctAnswer']]
        target_sentences = ['Was soll sie tun, die tschechischen Sozialdemokraten in Prag kennen weder Voldemort',
                            'Was soll sie tun, die tschechischen Sozialdemokraten in Prag kennen weder Voldemort']
        print(row['score'])

        # Convert sentences to token ids sequences.
        source_sentences_ids = [utils.sentence_to_token_ids_pandas(sent, source_vocab, FLAGS.max_seq_length)
                                for sent in source_sentences]
        target_sentences_ids = [utils.sentence_to_token_ids(sent, target_vocab, FLAGS.max_seq_length)
                                for sent in target_sentences]

        # Do stuff
        pairs = [(i, j) for i, j in zip(range(len(source_sentences)),
                                        range(len(target_sentences)))]

        data = [(source_sentences_ids[i], target_sentences_ids[j], 1.0)
                for i, j in zip(range(len(source_sentences)),
                                range(len(target_sentences)))]

        data_iterator = utils.TestingIterator(np.array(data, dtype=object))

        # Do more stuff
        x_source, source_seq_length,\
        x_target, target_seq_length,\
        labels = placeholders

        num_iter = int(np.ceil(data_iterator.size / FLAGS.batch_size))
        probs = []
        for step in xrange(num_iter):
            source, target, label = data_iterator.next_batch(FLAGS.batch_size)
            source_len = utils.sequence_length(source)
            target_len = utils.sequence_length(target)

            feed_dict = {x_source: source,
                         x_target: target,
                         labels: label,
                         source_seq_length: source_len,
                         target_seq_length: target_len}

            batch_probs, source_final_state_ph_please = sess.run([probs_op, source_final_state_ph], feed_dict=feed_dict)
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
            b = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_b")
            normalize_a = tf.nn.l2_normalize(a, 0)
            normalize_b = tf.nn.l2_normalize(b, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            sess2 = tf.Session()
            cos_sim = sess2.run(cos_similarity, feed_dict={a: source_final_state_ph_please[0], b: source_final_state_ph_please[1]})
            print("please", cos_sim)
            probs.extend(batch_probs.tolist())
        probs = np.array(probs[:data_iterator.size])
    return probs


def extract_pairs(sess, source_path, target_path,
                  source_vocab, target_vocab,
                  probs_op, placeholders, source_final_state_ph):
    """Extract sentence pairs from a pair of articles in source and target languages.
       Returns a list of (source sentence, target sentence, probability score) tuples.
    """

    y_score = inference(sess, source_path, target_path, source_vocab, target_vocab, probs_op, placeholders, source_final_state_ph)


def main(_):
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required."
    assert FLAGS.extract_dir, "--extract_dir is required."
    assert FLAGS.source_vocab_path, "--source_vocab_path is required."
    assert FLAGS.target_vocab_path, "--target_vocab_path is required."
    assert FLAGS.source_output_path, "--source_output_path is required."
    assert FLAGS.target_output_path, "--target_output_path is required."
    assert FLAGS.score_output_path, "--score_output_path is required."
    assert FLAGS.source_language, "--source_language is required."
    assert FLAGS.target_language, "--target_language is required."

    # Read vocabularies.
    source_vocab, _ = utils.initialize_vocabulary(FLAGS.source_vocab_path)
    target_vocab, _ = utils.initialize_vocabulary(FLAGS.target_vocab_path)

    # Read source and target paths for sentence extraction.
    source_paths = []
    target_paths = []
    for file in os.listdir(FLAGS.extract_dir):
        if file.endswith(FLAGS.source_language):
            source_paths.append(os.path.join(FLAGS.extract_dir, file))
        elif file.endswith(FLAGS.target_language):
            target_paths.append(os.path.join(FLAGS.extract_dir, file))
    source_paths.sort()
    target_paths.sort()

    utils.reset_graph()
    with tf.Session() as sess:
        # Restore saved model.
        utils.restore_model(sess, FLAGS.checkpoint_dir)

        # Recover placeholders and ops for extraction.
        x_source = sess.graph.get_tensor_by_name("x_source:0")
        source_seq_length = sess.graph.get_tensor_by_name("source_seq_length:0")

        x_target = sess.graph.get_tensor_by_name("x_target:0")
        target_seq_length = sess.graph.get_tensor_by_name("target_seq_length:0")

        labels = sess.graph.get_tensor_by_name("labels:0")

        placeholders = [x_source, source_seq_length, x_target, target_seq_length, labels]

        probs = sess.graph.get_tensor_by_name("feed_forward/output/probs:0")

        source_final_state_ph = sess.graph.get_tensor_by_name("birnn/source_final_state_ph:0")

        with open(FLAGS.source_output_path, mode="w", encoding="utf-8") as source_output_file,\
             open(FLAGS.target_output_path, mode="w", encoding="utf-8") as target_output_file,\
             open(FLAGS.score_output_path, mode="w", encoding="utf-8") as score_output_file:

            for source_path, target_path in zip(source_paths, target_paths):
                # Extract sentence pairs.
                extract_pairs(sess, source_path, target_path,
                                      source_vocab, target_vocab,
                                      probs, placeholders, source_final_state_ph)


if __name__ == "__main__":
    tf.app.run()