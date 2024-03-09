from bpe import BpeTokenizer
from kmer import KmerTokenizer
from unigram import UnigramTokenizer
import argparse
from pathlib import Path
import json
from utils import load_sequences, plot_and_save_evaluation_results, calculate_token_statistics

def main():
    parser = argparse.ArgumentParser(description="Command line interface for tokenizer creation")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer")
    parser.add_argument("--tokenizer-type", type=str, required=True, help="Type of tokenizer")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--evaluate-dir", type=str, required=True, help="Evaluation directory")
    parser.add_argument("--limit-files", type=int, default=10, help="Number of files to read in.")
    parser.add_argument("--samples-per-file", type=int, default=100, help="Number of sequences to sample from each file")
    parser.add_argument("--k", type=int, default=3, help="K-mer length")
    parser.add_argument("--overlap", action='store_true', help="Generate overlapping k-mers (default)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    input_dir = Path(args.input_dir)
    evaluate_dir = Path(args.evaluate_dir)

    sequences = load_sequences(
        input_dir, 
        limit_files=args.limit_files, 
        samples_per_file=args.samples_per_file, 
        random_files=True
    )

    print("Load evaluation data")
    evaluation_data = load_sequences(evaluate_dir,limit_files=10,samples_per_file=1000)

    # Define a range of vocabulary sizes to explore
    vocab_sizes = [16000, 24000, 32000, 48000, 64000]

    # Dictionary to store evaluation results for each vocab size
    evaluation_results = {}

    for vocab_size in vocab_sizes:
        print(f"Creating and evaluating tokenizer for vocab size: {vocab_size}")
        tokenizer_name=args.tokenizer_name+"_"+str(vocab_size)+"_"+str(args.limit_files)+"_"+str(args.samples_per_file)
        if args.overlap:
            tokenizer_type = args.tokenizer_type +"_" + str(args.k) + "_overlap"
        else:
            tokenizer_type = args.tokenizer_type +"_" + str(args.k) + "_nooverlap"
        
        tokenizer_path=str((output_dir / f"{tokenizer_type}_{tokenizer_name}.json").resolve())
        if Path(tokenizer_path).exists():
            tokenizer=Tokenizer.from_file(tokenizer_path)
        else:
            if args.tokenizer_type == "BPE":
                tokenizer = BpeTokenizer(vocab_size=vocab_size)
                tokenizer.train(sequences)
                tokenizer.save(output_dir, tokenizer_name)
            elif args.tokenizer_type == "Unigram":
                tokenizer = UnigramTokenizer(vocab_size=vocab_size)
                tokenizer.train(sequences)
                tokenizer.save(output_dir, tokenizer_name)
            elif args.tokenizer_type == "Kmer":
                tokenizer = KmerTokenizer(args.k, args.overlap)
                tokenizer.build_vocab(sequences)
                print(tokenizer.vocab)
                tokenizer.save(output_dir, tokenizer_name)
            else:
                raise ValueError("Unsupported tokenizer type. Choose 'BPE' or 'Unigram' or 'kmer'.")
        
        print("Evaluate tokenizer...")
        tokenized_data = tokenizer.encode_in_batches(evaluation_data)
        token_statistics = calculate_token_statistics(tokenized_data)
        evaluation_results[vocab_size] = token_statistics

     # Print evaluation results for each vocab size
#    for vocab_size, results in evaluation_results.items():
#        print(f"Vocab Size: {vocab_size}, Evaluation Results: {results}")

    output_file_path = str((output_dir / f"evaluation_results_{tokenizer_name}_{args.tokenizer_type}.json").resolve())
    
    with open(output_file_path, 'w') as file:
        json.dump(evaluation_results, file, indent=4)

    plot_and_save_evaluation_results(evaluation_results, output_dir, tokenizer_type + "_" + tokenizer_name)


if __name__ == "__main__":
    main()
    
