    n = 7178
    k = 10
    p = 0.3
    G = nx.newman_watts_strogatz_graph(n, k, p, directed=True)

    # Aggiunta di pesi casuali negli archi
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(low=-1, high=0)

    # Tracciamento della distribuzione di grado in entrata e in uscita
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist(in_degrees, bins=100, alpha=0.5, label='In Degree')
    plt.hist(out_degrees, bins=100, alpha=0.5, label='Out Degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.legend(loc='best')
    plt.show()