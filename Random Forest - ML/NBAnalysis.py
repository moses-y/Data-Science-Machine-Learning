def nb_analysis(nb_path):
    """Analyzes notebook, processing both code and markdown cells."""
    local_summaries = []  # Local list to store structured summaries for the current notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        all_cells = nb["cells"]
        print(f"Notebook: {os.path.basename(nb_path)} - Total cells: {len(all_cells)}")

        for i, cell in enumerate(all_cells):
            cell_type = cell["cell_type"]
            content = cell["source"]
            inputs = tokenizer.encode_plus(
                content,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="max_length",
            )
            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                do_sample=True,  # Enable sampling-based generation
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Structure each summary as a dictionary
            local_summaries.append(
                {"type": cell_type, "summary": summary, "index": i + 1}
            )

    global_summaries.extend(local_summaries)
