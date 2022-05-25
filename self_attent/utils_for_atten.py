import tqdm
from nltk.translate.bleu_score import corpus_bleu
import torch

def evaluate(model, iterator, criterion, TRG_vocab):
    model.eval()
    original_text = []
    generated_text = []
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm_notebook(iterator)):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])

            trg_pred = output.argmax(dim=2).cpu().numpy()
            orig = trg.cpu().numpy()
            for orig_txt, gen_txt in zip(orig, trg_pred):
                original_text.append(get_text(orig_txt, TRG_vocab))
                generated_text.append(get_text(gen_txt, TRG_vocab))

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
    return epoch_loss / len(iterator), bleu


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    history = []
    for i, batch in enumerate(tqdm.tqdm_notebook(iterator)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        history.append(loss.cpu().data.numpy())
    return epoch_loss / len(iterator)


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def bad_and_good_translations(model, iterator, criterion, TRG_vocab):
    model.eval()
    epoch_losses = []
    list_of_original = []
    list_of_generated = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm_notebook(iterator)):
            src = batch.src
            trg = batch.trg
            output, _ = model(src, trg[:, :-1])

            trg_pred = output.argmax(dim=2).cpu().numpy()
            orig = trg.cpu().numpy()
            for orig_txt, gen_txt in zip(orig, trg_pred):
                list_of_original.append(get_text(orig_txt, TRG_vocab))
                list_of_generated.append(get_text(gen_txt, TRG_vocab))

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_losses.append(loss.item())
    return epoch_losses, list_of_original, list_of_generated