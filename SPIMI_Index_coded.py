#-------------------------------------------------------------
# AUTHOR: Brian Tang
# FILENAME: SPIMI_index.py
# SPECIFICATION: SPIMI-based inverted index construction
# FOR: CS 5180 - Assignment #2
# TIME SPENT: 3 hours
#-------------------------------------------------------------

import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10

READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


# ---------------------------------------------------------
# 1–4) BUILD BLOCK INDEXES (SPIMI PHASE)
# ---------------------------------------------------------

chunks = pd.read_csv(
    INPUT_PATH,
    sep="\t",
    names=["doc_id", "text"],
    chunksize=BLOCK_SIZE,
    encoding="utf-8"
)

block_num = 1

for chunk in chunks:

    # convert docIDs like D0001 → 1
    chunk["doc_id"] = chunk["doc_id"].str.replace("D", "", regex=False).astype(int)

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunk["text"])
    terms = vectorizer.get_feature_names_out()

    partial_index = {}

    # build inverted index
    for doc_idx, doc_id in enumerate(chunk["doc_id"]):

        row = X.getrow(doc_idx)
        cols = row.indices

        for col in cols:
            term = terms[col]
            partial_index.setdefault(term, set()).add(doc_id)

    # write block file
    with open(f"block_{block_num}.txt", "w", encoding="utf-8") as f:

        for term in sorted(partial_index.keys()):
            postings = sorted(partial_index[term])
            line = f"{term}:{','.join(map(str, postings))}\n"
            f.write(line)

    block_num += 1


# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------

block_files = [
    open(f"block_{i}.txt", "r", encoding="utf-8")
    for i in range(1, NUM_BLOCKS + 1)
]

# buffers
read_buffers = [[] for _ in range(NUM_BLOCKS)]
buffer_pointers = [0] * NUM_BLOCKS

# load initial buffers
for i, f in enumerate(block_files):

    for _ in range(READ_BUFFER_LINES_PER_FILE):
        line = f.readline()
        if not line:
            break
        read_buffers[i].append(line.strip())


# ---------------------------------------------------------
# Helper: load next buffer chunk if needed
# ---------------------------------------------------------

def refill_buffer(file_index):

    if buffer_pointers[file_index] >= len(read_buffers[file_index]):

        read_buffers[file_index] = []
        buffer_pointers[file_index] = 0

        for _ in range(READ_BUFFER_LINES_PER_FILE):
            line = block_files[file_index].readline()
            if not line:
                break
            read_buffers[file_index].append(line.strip())


# ---------------------------------------------------------
# 7) INITIALIZE HEAP
# ---------------------------------------------------------

heap = []

for i in range(NUM_BLOCKS):
    if read_buffers[i]:
        term = read_buffers[i][0].split(":")[0]
        heapq.heappush(heap, (term, i))


# ---------------------------------------------------------
# 8–9) MERGE LOOP + WRITE BUFFER
# ---------------------------------------------------------

write_buffer = []

with open("final_index.txt", "w", encoding="utf-8") as out:

    while heap:

        current_term, file_idx = heapq.heappop(heap)

        merged_postings = set()

        files_to_process = [(current_term, file_idx)]

        # collect all identical terms from heap
        while heap and heap[0][0] == current_term:
            files_to_process.append(heapq.heappop(heap))

        # process all blocks with this term
        for _, idx in files_to_process:

            line = read_buffers[idx][buffer_pointers[idx]]
            postings = line.split(":")[1].split(",")

            merged_postings.update(map(int, postings))

            buffer_pointers[idx] += 1

            refill_buffer(idx)

            if buffer_pointers[idx] < len(read_buffers[idx]):
                next_term = read_buffers[idx][buffer_pointers[idx]].split(":")[0]
                heapq.heappush(heap, (next_term, idx))

        merged_postings = sorted(merged_postings)

        output_line = f"{current_term}:{','.join(map(str, merged_postings))}\n"
        write_buffer.append(output_line)

        # flush write buffer
        if len(write_buffer) >= WRITE_BUFFER_LINES:
            out.writelines(write_buffer)
            write_buffer = []

    # final flush
    if write_buffer:
        out.writelines(write_buffer)


# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------

for f in block_files:
    f.close()