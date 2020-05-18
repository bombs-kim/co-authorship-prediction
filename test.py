
def test(embedding_model, classifier, device, filepath='query_private.txt', savepath='answer_private.txt'):
    f = open(filepath, 'r')
    lines = f.readlines() 
    wf = open(savepath, 'w')

    for line in lines[1:]:
        node = line.strip().split(' ')
        node = [int(i) for i in node]
        node = torch.tensor(node, dtype=torch.long)
        
        with torch.no_grad():
            embed = embedding_model(node.to(device))
            score = classifier(embed.unsqueeze(1))

        if score.item() > 0.5:
            pred = 'True'
        else:
            pred = 'False'

        wf.write(pred + '\n')

    wf.close()
