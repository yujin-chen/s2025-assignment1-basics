import time
from BPE_Tokenizer import BPE_Tokenizer  

def test_tokenizer(tokenizer, sample_texts):
    start_time = time.time()  # Start timing
    for i, text in enumerate(sample_texts):
        token_ids = tokenizer.encode(text)
        byte_len = len(text.encode("utf-8"))
        num_tokens = len(token_ids)
        comp_ratio = byte_len / num_tokens if num_tokens else 0
        print(f"Sample {i+1}:")
        print(f"Text: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Compression ratio (bytes/token): {comp_ratio:.2f}")
        print("-" * 40)
    end_time = time.time()  # End timing
    print(f"Tokenization time: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    tiny_vocab_path = "Tiny_bpe_vocab1.pkl"         # 
    tiny_merges_path = "Tiny_bpe_merges1.pkl"          
    open_vocab_path = "real_owt_bpe_vocab1.pkl"         
    open_merges_path = "real_owt_bpe_merges1.pkl"          

    special_tokens = ["<|endoftext|>"]

    # Load tokenizers using the class method from_files:
    print("Loading TinyStories Tokenizer...")
    tiny_tokenizer = BPE_Tokenizer.from_files(tiny_vocab_path, tiny_merges_path, special_tokens)
    print("Loading OpenWebText Tokenizer...")
    open_tokenizer = BPE_Tokenizer.from_files(open_vocab_path, open_merges_path, special_tokens)

    # (a) Sample 10 documents from each dataset.
    tiny_samples = [
        "One day, a little boy named Tim went to the park. He saw a big tiger. The tiger was not mean, but very easy to play with. Tim and the tiger played all day. They had lots of fun.",
        "Sara and Ben wanted to decorate a bowl for their mom. They found a big bowl in the kitchen and some paint and brushes. They took the bowl and the paint to the backyard and put them on a table.",
        "But then, something unexpected happened. A big dog came and took the toy car in its mouth! Tim and Sam were scared, but the dog just wanted to play too. They all played together, and the dog was very gentle with the car. In the end, Tim, Sam, and the dog became good friends.\n<|endoftext|>",
        "Once upon a time, there was a sailor named Tom. Tom had a big boat. He liked to sail on the sea. One day, Tom saw a little fish. The fish was sad. It was lost and wanted to go home.",
        "One day, a little girl named Lucy went to the park with her mom. They saw a big tree with lots of flowers. The flowers had sweet smells. Lucy liked the smells. She wanted to count the flowers on the tree.\n\"Mom, can I count the flowers?\" Lucy asked. Her mom said, \"Yes, you can count the flowers. Be careful not to touch them.\"",
        "One day, the hunter and Spot went into the woods to find food. They saw a big bird in a tree. The hunter got ready to shoot. Spot barked loudly. The bird flew away. They did not catch the bird, but they were not sad.\nThe hunter and Spot played in the woods. They ran and jumped. They had lots of fun. Then, they went back to their small house. They were happy to be together. And they lived happily ever after.\n<|endoftext|>",
        "After they fixed the car, they decided to paint it. They painted it with many colors. When they were done, the car looked very pretty. But then, something unexpected happened. The car started to move on its own! Lily and her dad were very surprised. They had so much fun playing with the magical toy car.",
        "Her mom said, \"Don't worry, honey. We'll figure it out together.\"\nThey went to the garden and worked to fix it. They watered the flowers and got rid of the bugs. Soon, the garden was beautiful again.",
        "Tim went inside the toilet and tried to use it. But, suddenly, the toilet broke and water came out. Tim got very wet and dirty. He cried and said, \"Mom, I don't like this toilet!\" His mom hugged him and said, \"I know, Tim. We will find a better one next time.\" They went home with Tim still feeling sad and wet.",
        "Once upon a time there was a person named Sue. She weighed a lot and her momma said that she had to start eating less and exercising more. So, Sue decided to start walking around the block every day."
    ]

    owt_samples = [
        "When They Come Calling doesn't rely on paranormal hooks: it’s not werewolves, vampires, zombies, or nymphomaniacs engaged in magic or erotic adventures, but instead a classic tale of love and suspense; a modernized ghost story of two lost souls, drawn together until fate tears them apart.",
        "Jed is a warrior from another era, haunted by the horrors of a brutal family feud three thousand years in the making, and inspired by his own secret quest. Relentless and driven, Jed’s determination radiates to everyone around him.",
        "You'll need an HTML5 capable browser to see this content. Play Replay with sound Play with",
        "Anna is a physician from Kansas City. She’s spent her life giving and caring to others while trying to hide how different she is. Anna has lost everyone she’s ever loved: her relationships, her family, and her hope for something more.",
        "The base funding will allow us to produce a professional book. There are three areas that the minimum funding will go towards to ensure we create a quality product::",
        "$3500 -- Every backer will be able to send a free gift copy of the ebook to a friend. (at this level of funding, we will be able to get a higher quality cover designed)",
        "\"We can therefore imagine that those who have 1,000 or 2,000 friends on Facebook may be subjected to even greater stress,\" said Lupien.",
        "ⓒ 2018 TECHTIMES.com All rights reserved. Do not reproduce without permission.<|endoftext|>Google’s Pixel is no doubt a hit, with some models still out of stock a couple of months after launch. An analysis by Morgan Stanley gives us an even more detailed picture about just how well the phone is doing and what it may mean for Google’s place in the smartphone market.",
        "And not only did Morneau not really answer the question, he even seemed somewhat indignant that he had to answer to the public in the first place.",
        "There was certainly the appearance of a conflict, or at least the appearance of a risk of conflict, to have someone in such a powerful post hold millions of dollars in a company that stands to gain from decisions made by said person."
    ]

    print("\n--- (a) Encoding 10 TinyStories Samples ---")
    test_tokenizer(tiny_tokenizer, tiny_samples)

    print("\n--- (a) Encoding 10 OpenWebText Samples ---")
    test_tokenizer(open_tokenizer, owt_samples)

    # Encode the OpenWebText samples with the TinyStories tokenizer.
    print("\n--- (b) Tokenizing OpenWebText Samples with TinyStories Tokenizer ---")
    test_tokenizer(tiny_tokenizer, owt_samples)

    #Estimate throughput:
    large_sample = " ".join(owt_samples * 1000)
    start = time.time()
    _ = tiny_tokenizer.encode(large_sample)
    elapsed = time.time() - start
    byte_size = len(large_sample.encode("utf-8"))
    throughput = byte_size / elapsed
    print(f"\n---rough estimate of throughput ---\n")
    print(f"Throughput: {throughput:.2f} bytes/sec")
    total_bytes = 825 * (1024**3)
    estimated_time_sec = total_bytes / throughput
    estimated_time_hours = estimated_time_sec / 3600
    print(f"Estimated time to tokenize 825GB: {estimated_time_hours:.2f} hours")
