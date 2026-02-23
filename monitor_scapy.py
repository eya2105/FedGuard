from scapy.all import sniff, IP, TCP, UDP, ICMP
from collections import defaultdict, deque
import time
import statistics
import math

# configuration 
window_size = 1 # taille de la fenetre ( en seconde )
iface = "s1-eth1" # interface à surveiller  (c'est un interface mininet )
N_HISTORY = 5  # Nombre de fenêtres pour calculer les std inter-fenêtres
# Mémoire des paquets par flux 5-tuple (src_ip, src_port, dst_ip, dst_port, proto)
# et aussi index pour accéder aux stats 4-tuple
flows_5tuple = defaultdict(lambda: deque()) # Stockage par connexion réelle
flows_4tuple = defaultdict(lambda: {
    'paquets': [], # Pour garder trace si besoin
    'connexions': set() # Pour savoir combien de connexions actives 
})
print(f"=== Surveillance temps réel ({window_size}s) avec Scapy ===")
print("-" * 70)

# Fonction pour identifier un flux (5-tuple pour stockage)

def get_flow_key_5tuple(pkt):
    if IP in pkt:
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
       
        src_port = 0
        dst_port = 0
        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
           
        return (src_ip, src_port, dst_ip, dst_port, proto)
    return None

# Fonction pour obtenir la clé 4-tuple à partir de la clé 5-tuple

def get_4tuple_from_5tuple(key_5tuple):
    src_ip, src_port, dst_ip, dst_port, proto = key_5tuple
    return (src_ip, dst_ip, dst_port, proto)

# Callback appelé à chaque paquet

def process_packet(pkt):
    ts = pkt.time
    key_5tuple = get_flow_key_5tuple(pkt)
    if key_5tuple:
        # Calculer la taille exacte de l'en-tête
        taille_entete_ip = pkt[IP].ihl * 4 if IP in pkt else 0
        taille_entete_l4 = 0
        proto = key_5tuple[4]
        if proto == 6 and TCP in pkt:  # TCP
            taille_entete_l4 = pkt[TCP].dataofs * 4
        elif proto == 17 and UDP in pkt:  # UDP
            taille_entete_l4 = 8
        elif proto == 1 and ICMP in pkt:  # ICMP
            taille_entete_l4 = 8
        # Pour les autres protocoles, estimation (juste IP)
       
        taille_entete = taille_entete_ip + taille_entete_l4
       
        # Déterminer si fragmenté
        is_frag = (IP in pkt and ('MF' in pkt[IP].flags or pkt[IP].frag > 0))
       
        # Stocker dans le flux 5-tuple (connexion réelle)
        flows_5tuple[key_5tuple].append((ts, len(pkt), taille_entete, is_frag))
       
        # Mettre à jour l'index 4-tuple
        key_4tuple = get_4tuple_from_5tuple(key_5tuple)
        flows_4tuple[key_4tuple]['connexions'].add(key_5tuple)

# Fonction pour calculer l'IAT moyen d'une connexion (5-tuple)

def calculer_iat_connexion(pkt_list, current_time):
    # Filtrer les paquets de la fenêtre
    window_packets = [(ts, size, entete, frag) for ts, size, entete, frag in pkt_list if ts >= current_time - window_size]
   
    if len(window_packets) <= 1:
        return None # Pas assez de paquets pour calculer IAT
   
    timestamps, _, _, _ = zip(*window_packets)
    iats = [(timestamps[i] - timestamps[i-1])*1000 for i in range(1, len(window_packets))]
   
    return {
        'moyenne': statistics.mean(iats),
        'ecart_type': statistics.stdev(iats) if len(iats) > 1 else 0.0,
        'nb_paquets': len(window_packets),
        'nb_iats': len(iats)
    }

# Fonction pour afficher stats

def print_stats():
    current_time = time.time()
    any_flow = False
   
    # Parcourir tous les flux 4-tuple
    for key_4tuple, info in flows_4tuple.items():
        if 'history' not in info:
            info['history'] = {
                'pkts_total': deque(maxlen=N_HISTORY),
                'bytes_total': deque(maxlen=N_HISTORY),
                'pkt_size_avg': deque(maxlen=N_HISTORY),
                'pkt_size_std': deque(maxlen=N_HISTORY),
                'pkt_arrivals_avg': deque(maxlen=N_HISTORY),
                'port_src_unique': deque(maxlen=N_HISTORY),
                'port_src_entropy': deque(maxlen=N_HISTORY),
                'conn_pkts_avg': deque(maxlen=N_HISTORY),
                'pkts_frag_share': deque(maxlen=N_HISTORY),
                'hdrs_payload_ratio_avg': deque(maxlen=N_HISTORY),
                'dominant_proto_ratio': deque(maxlen=N_HISTORY),
            }
       
        connexions_actives = []
        tous_iats = []
        total_paquets = 0
        total_bytes = 0
       
        # Liste pour collecter TOUTES les tailles de paquets du 4-tuple
        toutes_les_tailles = []
       
        # Compteurs pour les protocoles
        compteur_tcp = 0
        compteur_udp = 0
        compteur_icmp = 0
        compteur_autres = 0
       
        # Dictionnaire pour compter les occurrences de chaque port source
        compteur_ports = defaultdict(int)
       
        # Compteurs pour le calcul du ratio en-tête/payload
        total_entete = 0
        total_payload = 0
       
        # Compteur pour fragments
        compteur_fragments = 0
       
        # Examiner chaque connexion (5-tuple) de ce flux 4-tuple
        for key_5tuple in info['connexions']:
            if key_5tuple in flows_5tuple:
                pkt_list = flows_5tuple[key_5tuple]
               
                # Compter les paquets et bytes pour ce 5-tuple dans la fenêtre
                window_packets = [(ts, size, entete, frag) for ts, size, entete, frag in pkt_list if ts >= current_time - window_size]
                if window_packets:
                    connexions_actives.append(key_5tuple)
                    total_paquets += len(window_packets)
                   
                    # Récupérer le protocole
                    proto = key_5tuple[4]
                   
                    #Pour chaque paquet, calculer en-tête et payload
                    for _, taille_totale, taille_entete, is_frag in window_packets:
                        toutes_les_tailles.append(taille_totale)
                        total_bytes += taille_totale
                       
                        taille_payload = max(0, taille_totale - taille_entete)
                       
                        total_entete += taille_entete
                        total_payload += taille_payload
                       
                        if is_frag:
                            compteur_fragments += 1
                   
                    # Compter les paquets par port source
                    src_port = key_5tuple[1]
                    nb_paquets_connexion = len(window_packets)
                    compteur_ports[src_port] += nb_paquets_connexion
                   
                    # Compter les paquets par protocole
                    if proto == 6: # TCP
                        compteur_tcp += nb_paquets_connexion
                    elif proto == 17: # UDP
                        compteur_udp += nb_paquets_connexion
                    elif proto == 1: # ICMP
                        compteur_icmp += nb_paquets_connexion
                    else:
                        compteur_autres += nb_paquets_connexion
                   
                    # Calculer IAT pour cette connexion
                    iat_info = calculer_iat_connexion(pkt_list, current_time)
                    if iat_info:
                        tous_iats.append(iat_info)
       
        if not connexions_actives:
            continue
           
        any_flow = True
       
        # Calculer les stats globales pour le 4-tuple
        packets_window = total_paquets
        bytes_window = total_bytes
        pkt_rate = packets_window / window_size
        byte_rate = bytes_window / window_size
       
        # Calcul de l'entropie des ports sources
        if packets_window > 0 and compteur_ports:
            entropie_ports = 0.0
            for count in compteur_ports.values():
                probabilite = count / packets_window
                entropie_ports -= probabilite * math.log2(probabilite)
        else:
            entropie_ports = 0.0
       
        # Calcul des proportions
        if packets_window > 0:
            proportion_tcp = (compteur_tcp / packets_window) * 100
            proportion_udp = (compteur_udp / packets_window) * 100
            proportion_icmp = (compteur_icmp / packets_window) * 100
            proportion_autres = (compteur_autres / packets_window) * 100
        else:
            proportion_tcp = proportion_udp = proportion_icmp = proportion_autres = 0.0
       
        # Calcul de la taille moyenne et écart-type
        if toutes_les_tailles:
            taille_moyenne = sum(toutes_les_tailles) / len(toutes_les_tailles)
           
            if len(toutes_les_tailles) > 1:
                # Écart-type (formule avec n-1 pour échantillon)
                variance = sum((s - taille_moyenne)**2 for s in toutes_les_tailles) / (len(toutes_les_tailles) - 1)
                taille_std = variance ** 0.5
            else:
                taille_std = 0.0
        else:
            taille_moyenne = 0.0
            taille_std = 0.0
       
        # Calcul du ratio en-tête/payload
        if total_payload > 0:
            ratio_entete_payload = total_entete / total_payload
            # En pourcentage
            pourcentage_entete = (total_entete / (total_entete + total_payload)) * 100
            pourcentage_payload = (total_payload / (total_entete + total_payload)) * 100
        else:
            ratio_entete_payload = 0.0
            pourcentage_entete = 0.0
            pourcentage_payload = 0.0
       
        # Calculer l'IAT moyen pondéré par le nombre d'intervalles
        if tous_iats:
            # Moyenne pondérée des IAT (poids = nombre d'intervalles)
            total_iats = sum(iat['nb_iats'] for iat in tous_iats)
            iat_avg = sum(iat['moyenne'] * iat['nb_iats'] for iat in tous_iats) / total_iats if total_iats > 0 else 0.0
           
            # Écart-type global (approché)
            if len(tous_iats) > 1:
                # Variance pondérée
                variance_ponderee = sum(iat['ecart_type']**2 * iat['nb_iats'] for iat in tous_iats) / total_iats
                iat_std = variance_ponderee ** 0.5
            else:
                iat_std = tous_iats[0]['ecart_type'] if tous_iats else 0.0
        else:
            iat_avg = iat_std = 0.0
       
        # Features supplémentaires pour std inter-fenêtres
        port_src_unique = len(connexions_actives)
        conn_pkts_avg = total_paquets / port_src_unique if port_src_unique > 0 else 0.0
        pkts_frag_share = compteur_fragments / total_paquets if total_paquets > 0 else 0.0
        dominant_proto_ratio = max(proportion_tcp, proportion_udp, proportion_icmp, proportion_autres)
       
        # Ajouter aux historiques (seulement si flux actif cette fenêtre)
        history = info['history']
        history['pkts_total'].append(packets_window)
        history['bytes_total'].append(bytes_window)
        history['pkt_size_avg'].append(taille_moyenne)
        history['pkt_size_std'].append(taille_std)
        history['pkt_arrivals_avg'].append(iat_avg)
        history['port_src_unique'].append(port_src_unique)
        history['port_src_entropy'].append(entropie_ports)
        history['conn_pkts_avg'].append(conn_pkts_avg)
        history['pkts_frag_share'].append(pkts_frag_share)
        history['hdrs_payload_ratio_avg'].append(ratio_entete_payload)
        history['dominant_proto_ratio'].append(dominant_proto_ratio)
       
        # Calculer les std inter-fenêtres
        def calc_std(hist):
            if len(hist) > 1:
                return statistics.stdev(hist)
            return 0.0
       
        pkts_total_std = calc_std(history['pkts_total'])
        bytes_total_std = calc_std(history['bytes_total'])
        pkt_size_avg_std = calc_std(history['pkt_size_avg'])
        pkt_size_std_std = calc_std(history['pkt_size_std'])
        pkt_arrivals_avg_std = calc_std(history['pkt_arrivals_avg'])
        port_src_unique_std = calc_std(history['port_src_unique'])
        port_src_entropy_std = calc_std(history['port_src_entropy'])
        conn_pkts_avg_std = calc_std(history['conn_pkts_avg'])
        pkts_frag_share_std = calc_std(history['pkts_frag_share'])
        hdrs_payload_ratio_avg_std = calc_std(history['hdrs_payload_ratio_avg'])
        dominant_proto_ratio_std = calc_std(history['dominant_proto_ratio'])
       
        src_ip, dst_ip, dst_port, proto = key_4tuple
        print(f"Flux: {src_ip} -> {dst_ip}:{dst_port} (Proto: {proto})")
        print(f" >>> nombre de ports source distincts : {len(connexions_actives)}")
        print(f" >>> Paquets dans fenêtre : {packets_window}")
        print(f" >>> Bytes dans fenêtre : {bytes_window}")
        print(f" >>> Packet rate : {pkt_rate:.2f} pps")
        print(f" >>> Byte rate : {byte_rate:.2f} Bps")
        print(f" >>> Entropie ports source : {entropie_ports:.3f} bits")
        print(f" >>> Proportion TCP : {proportion_tcp:.1f}%")
        print(f" >>> Proportion UDP : {proportion_udp:.1f}%")
        print(f" >>> Proportion ICMP : {proportion_icmp:.1f}%")
        if proportion_autres > 0:
            print(f" >>> Proportion autres : {proportion_autres:.1f}%")
        print(f" >>> Ratio en-tête/payload : {ratio_entete_payload:.3f}")
        print(f" >>> └─ % en-tête : {pourcentage_entete:.1f}%")
        print(f" >>> └─ % payload : {pourcentage_payload:.1f}%")
        print(f" >>> IAT moyen (pondéré) : {iat_avg:.2f} ms")
        print(f" >>> IAT stddev : {iat_std:.2f} ms")
        print(f" >>> Taille moyenne paquet : {taille_moyenne:.2f} octets")
        print(f" >>> Écart-type taille : {taille_std:.2f} octets")
        print("-" * 70)
        # Afficher les std inter-fenêtres
        print(f" >>> pkts_total_std : {pkts_total_std:.2f}")
        print(f" >>> bytes_total_std : {bytes_total_std:.2f}")
        print(f" >>> pkt_size_avg_std : {pkt_size_avg_std:.2f}")
        print(f" >>> pkt_size_std_std : {pkt_size_std_std:.2f}")
        print(f" >>> pkt_arrivals_avg_std : {pkt_arrivals_avg_std:.2f}")
        print(f" >>> port_src_unique_std : {port_src_unique_std:.2f}")
        print(f" >>> port_src_entropy_std : {port_src_entropy_std:.2f}")
        print(f" >>> conn_pkts_avg_std : {conn_pkts_avg_std:.2f}")
        print(f" >>> pkts_frag_share_std : {pkts_frag_share_std:.2f}")
        print(f" >>> hdrs_payload_ratio_avg_std : {hdrs_payload_ratio_avg_std:.2f}")
        print(f" >>> dominant_proto_ratio_std : {dominant_proto_ratio_std:.2f}")
        print("-" * 70)
   
    if not any_flow:
        print(">>> Aucun flux actif dans cette fenêtre <<<")
        print("-" * 70)

# Boucle principale

last_print = time.time()
print(f"Démarrage de la capture sur {iface}...")
print("-" * 70)
while True:
    # Sniff par petits lots pour être réactif
    sniff(iface=iface, prn=process_packet, timeout=1, store=False)
   
    # Afficher les stats toutes les window_size secondes
    if time.time() - last_print >= window_size:
        print(f"\n[RAPPORT FENÊTRE {window_size} SECONDES - {time.strftime('%H:%M:%S')}]")
        print_stats()
        last_print = time.time()

