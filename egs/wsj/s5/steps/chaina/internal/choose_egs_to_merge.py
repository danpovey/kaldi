#!/usr/bin/env python3

# Copyright  2018  Johns Hopkins University (author: Daniel Povey)
# License: Apache 2.0.

import os
import argparse
import sys
import re




parser = argparse.ArgumentParser(description="Chooses groups of examples to merge into groups "
                                 "of size given by the --chunks-per-spk option, based on speaker "
                                 "information (preferentially, chunks from the same utterance "
                                 "and, if possible, the same speaker, get combined into "
                                 "groups).  This script also computes a held-out subset of...",
                                 epilog="E.g. " + sys.argv[0] + "*** TODO *** ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Also maybe have --num-repeats, which must divide --chunks-per-spk?  Can be
# used to divide data into different groups than the default ones.


parser.add_argument("--chunks-per-spk", type=int, default=4,
                    help="Number of chunks per speaker in the final egs (actually "
                    "means the number of chunks per group of chunks, and they are "
                    "only preferentially taken from the same speaker.")
parser.add_argument("--num-repeats", type=int, default=1,
                    "The number of times the data is to be repeated.  Must divide "
                    "--chunks-per-spk.  Suggest to try only 1 or 2.  The idea "
                    "is to divide chunks into groups in different ways, to give "
                    "more variety to the egs (since the adaptation information "
                    "will differ.")
parser.add_argument("--heldout-data-selection-proportion", type=float,
                    default=0.2,
                    "This parameter governs the selection of the heldout "
                    "subset and the statistically matched training subset. "
                    "It does not affect the size of that subset, but only "
                    "affects what pool the examples are drawb from.  "
                    "Smaller values of this mean that the heldout groups "
                    "will be preferentially drawn from groups that "
                    "'contaminate' the least number of other groups, "
                    "and so require the least data to be removed from the "
                    "training set.  Setting this to 1.0 would mean that "
                    "the heldout subset is drawn completely at random "
                    "(which might be more wasteful of training data, but "
                    "gives a selection that's statistically more "
                    "representative).")
parser.add_argument("--num-heldout-groups", type=int, default=200,
                    "Number of utterance groups "
                    "that will go in the heldout subset (and in the "
                    "statistically matched training subset)")
parser.add_argument("--utt2uniq", type=str, default='',
                    "File used in setups with data "
                    "augmentation, that maps from utterance-ids to the "
                    "pre-augmentation utterance-id.  The reason it's needed "
                    "is to ensure that the heldout set is properly held "
                    "out (i.e., that different versions of those utterances "
                    "weren't trained on.  If not specified, we assume the "
                    "identity map.")
parser.add_argument("--scp-in", type=str, required=True,
                    "The scp file in, likely containing chain egs.  The "
                    "keys are expected to be of the form: "
                    "'<utterance_id>-<first_frame>-<left_context>-<num_frames>-<right_context>-v1', "
                    "where the left_context, num_frames and right_context are required to be the "
                    "same in order for keys to be in a group (note: it's best if the "
                    "--extra-left-context-initial and --extra-right-context-final options "
                    "are not used, and if the --frames-per-chunk is a single number, in "
                    "order to prevent this constraint from splitting up the utterances from "
                    "a single speaker")
parser.add_argument("--training-data-out", type=str, required=True,
                    "The output file containing the chunks that are to be grouped; each "
                    "line will contain --chunks-per-spk (e.g. 4) rxfilenames, obtained "
                    "from the second field of the input --scp-in file.")
parser.add_argument("--heldout-subset-out", type=str, required=True,
                    "This is the name of the file to which the heldout data subset "
                    "will be written; the format is the same as --training-data-out.")
parser.add_argument("--training-subset-out", type=str, required=True,
                    "This is the name of the file to which the statistically matched "
                    "(to --heldout-subset-out) set of training data will be written")




"""
Notes on plan for how to implement this (we can keep this as documentation).
This is a rather simple plan and we might later implement something more
sophisticated that does a better job of keeping chunks from the same utterance
or speaker together.

It's pretty trivial.  Basically we rely on the fact that the input utterances
come in in sorted order (so utterances from adjacent speakers will naturally be
together.

We read the entries in the input scp file as a list, keeping them in the order
they were in the input.  We split that list into distinct sub-lists, each with a unique value
of <left_context>-<num_frames>-<right_context>, although in the expected
case there will be just one such sub-list.

In the case where --chunks-per-spk=4 and --num-repeats=1, the groups of
chunks would then just be (and we do this for each of the sub-lists):
the first 4 chunks; the second 4 chunks; and so on.  In the case where
--chunks-per-spk=4 and --num-repeats=2, we'd obtain the groups as above, then
we'd discard the first 2 chunks of each sub-list and repeat the process, giving
us twice the original number of groups.

Once we have the groups as above, we need to figure out the subset of
size --num-heldout-groups which will be chosen to appear in the output
file --heldout-subset-out.  We'll also be choosing another subset of
the same size to appear in the file --training-subset-out; and we'll
be excluding some groups from the output --training-data-out (any
utterances that appeared in --heldout-subset-out, or which were linked
with such utterances via the --utt2uniq map, will be excluded).

The way we choose the groups to hold out is as follows.  In cases where
the utt2uniq file is undefined, treat it as the identity map.
We are given list of groups.  We compute, for each group, the set of
utterances represented in it, and from that, the set of "uniq"
values (a "uniq" value is a string, representing a pre-augmentation
utterance-id).  For each "uniq" value, we compute the set of
group-ids in which it was represented.  For a given group, we
take the union of all those sets for its "uniq" value, and remove
its own group-id.  The size of this set gives us a number >= 0 of the
number of other groups we'd have to exclude if we were to include
this particular group in the heldout subset.  It might be zero only in
the case where there was no augmentation and --num-repeats=1, and
some particular utterance had been split into exactly 4 chunks which
all ended up in the same group.


num_groups which is
the number of groups in which that "uniq" value is represented.
Then, for each group, we can compute





"""
