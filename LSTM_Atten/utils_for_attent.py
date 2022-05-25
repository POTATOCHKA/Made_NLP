import torch
import tqdm


def flatten(l):
    return [item for sublist in l for item in sublist]


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


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


def bad_and_good_translations(model, iterator, criterion, TRG_vocab):
    model.eval()
    epoch_losses = []
    list_of_original = []
    list_of_generated = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm_notebook(iterator)):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)
            output_for_bleu = output.argmax(dim=-1)
            original = [get_text(x, TRG_vocab) for x in trg.cpu().numpy().T]
            generated = [get_text(x, TRG_vocab) for x in output_for_bleu[1:].detach().cpu().numpy().T]
            list_of_original.append(original)
            list_of_generated.append(generated)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)

            epoch_losses.append(loss.item())
    return epoch_losses, list_of_original, list_of_generated
