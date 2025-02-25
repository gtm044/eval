# Implement the following metrics:
# - Average Chunk Size
# - Jaccard Index
import tiktoken
import argparse

def mod(a: int):
    return a if a >= 0 else -a

def avg_chunk_size(contexts, limit=[3, 15]):
    """
    Calculate the average chunk size of the context documents and return a normalized index.
    Index ranges from -inf to 1.0 (Hgher is better)
    Any score above 0.5 is acceptable.
    """
    # Tokenize the context documents and get the token size of each document.
    sizes = 0
    enc = tiktoken.get_encoding("o200k_base")
    for context in contexts:
        num_tokens = len(enc.encode(context))
        sizes += num_tokens
    sizes = round(sizes / len(contexts))
    # Normalized parabolic function
    midpoint = (limit[0] + limit[1]) // 2
    index = 1-((2*mod(sizes-midpoint))/(limit[1]-limit[0]))
    
    return index

def jaccard_index(generated_chunks, reference_chunks):
    """
    Calculate the IoU (Jaccard Index) between the reference contexts and the retrived contexts.
    """
    # Tokenie the reference and generated chunks
    enc = tiktoken.get_encoding("o200k_base")
    ref_chunks, gen_chunks = [], []
    for ref_chunk in reference_chunks:
        ref_chunks.append(enc.encode(ref_chunk))
    for gen_chunk in generated_chunks:
        gen_chunks.append(enc.encode(gen_chunk))

    # Calculate the Jaccard Index
    jaccards = []
    for ref, gen in zip(ref_chunks, gen_chunks):
        intersection_result = list(set(ref).intersection(set(gen)))
        union_result = list(set(ref).union(set(gen)))
        jaccards.append(len(intersection_result) / len(union_result))
        
    return jaccards
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test", type=str)
    args = parser.parse_args()
    if args.test=="avg_chunk_size":
        contexts = [
            "This is a test document. It has a few sentences.",
            "This is another test document. It also has a few sentences.",
            "My name is Goutham"
        ]    
        print(avg_chunk_size(contexts))
    elif args.test=="jaccard_index":
        generated_chunks = [
            "This is a test mocument. It has a few sentences.",
            "This is another test document. It also has a few sentences."
        ]
        reference_chunks = [
            "This is a test document. It has a few sentences.",
            "This is another test document. It also has a few sentences.",
        ]
        print(jaccard_index(generated_chunks, reference_chunks))