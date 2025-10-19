// Includiamo la libraria standard di input/output
// Contiene operazioni base come printf e scanf
#include <stdio.h>
// This header file includes functions for memory allocation, process control, conversions, and others
// Contiene operazioni come malloc, free, exit
#include <stdlib.h>
// Includiamo la libraria matematica
// Contiene funzioni matematiche come sqrt, sin, cos, etc.
#include <math.h>
// Includiamo la libraria MPI per la programmazione parallela
// Contiene funzioni per la comunicazione tra processi in un ambiente di calcolo parallelo
// Alcuni esempi di funzioni sono MPI_Init, MPI_Comm_size, MPI_Comm_rank, MPI_Send, MPI_Recv
#include <mpi.h>
// Includiamo la libraria per la manipolazione delle stringhe
// Contiene funzioni per operazioni su stringhe come strlen, strcpy, strcat, strcmp
#include <string.h>
// Includiamo la libraria per la gestione del tempo
// Contiene funzioni per misurare il tempo e gestire operazioni temporali
// Alcuni esempi di funzioni sono gettimeofday, time, clock
#include <sys/time.h>
// Includiamo la libraria per operazioni di basso livello sul sistema operativo
// Contiene funzioni per operazioni come lettura/scrittura di file, gestione dei processi, manipolazione di file descriptor
// Alcuni esempi di funzioni sono read, write, fork, exec
#include <unistd.h>
// Includiamo la libraria per la gestione del tempo standard
// Contiene funzioni per operazioni temporali come time, difftime, mktime
#include <time.h>

// Definiamo la costante gravitazionale universale
#define G 6.67430e-11

// Definiamo il numero di corpi nel sistema
int NUM_BODIES = 1000;  // Default value
// Definiamo il numero di iterazioni per la simulazione
int NUM_STEPS = 100;  // Default value
// Definiamo il passo temporale per la simulazione
double DT = 1e4; // Passo temporale

// Definiamo la struttura per rappresentare un corpo celeste
// Contiene informazioni sulla massa, posizione e velocità del corpo
typedef struct {
    double mass;      // Massa del corpo
    double position[3]; // Posizione del corpo (x, y, z)
    double velocity[3]; // Velocità del corpo (vx, vy, vz)
} Body;

// Funzione per creare un tipo di dato MPI personalizzato per la struttura Body
MPI_Datatype create_body_datatype() {
    // Definiamo il tipo di dato MPI per la struttura Body
    MPI_Datatype body_type;
    // Definiamo gli array per i blocchi di lunghezza, gli spostamenti e i tipi di dato MPI
    // La struttura Body ha 3 campi: mass (double), position (array di 3 double), velocity (array di 3 double)
    int blocklengths[] = {1, 3, 3};
    
    // Calcoliamo gli spostamenti dei campi all'interno della struttura Body
    // MPI_Aint è un tipo di dato MPI usato per rappresentare gli offset in byte
    // Su sistemi a 32 bit: equivale a 4 byte
    // Su sistemi a 64 bit: equivale a 8 byte
    MPI_Aint offsets[3];
    // Calcoliamo gli spostamenti usando la macro offsetof
    // offsetof restituisce l'offset in byte di un membro all'interno di una struttura
    // Esempio numerico: 
    // Se la struttura Body inizia all'indirizzo 0x1000 e il campo mass si trova a 0x1000,
    // position a 0x1008 e velocity a 0x1020, allora offsets sarà {0, 8, 32}
    // Il calcolo avviene in questo modo: 
    // - mass: 0x1000 - 0x1000 = 0x0 = 0 (offset = 0)
    // - position: 0x1008 - 0x1000 = 0x8 = 8 (offset = 8) 
    // - velocity: 0x1020 - 0x1000 =  0x20 = 32 (offset = 32)
    // Layout memoria: mass(8 byte) + position(24 byte) + velocity(24 byte) = 56 byte totali    offsets[0] = offsetof(Body, mass);
    offsets[0] = offsetof(Body, mass);
    offsets[1] = offsetof(Body, position);
    offsets[2] = offsetof(Body, velocity);

    // Definiamo i tipi di dato MPI per ogni campo della struttura Body
    // mass è un double, position è un array di 3 double, velocity è un array di 3 double
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};

    // Creiamo il tipo di dato MPI per la struttura Body
    // Usando MPI_Type_create_struct per definire un tipo di dato composto
    // La funzione prende il numero di blocchi, gli array di lunghezze, spostamenti e tipi, e restituisce il nuovo tipo di dato MPI
    MPI_Type_create_struct(3, blocklengths, offsets, types, &body_type);
    // Commit del tipo di dato MPI per renderlo utilizzabile nelle comunicazioni
    MPI_Type_commit(&body_type);

    return body_type;
    // Creare il tipo di dato MPI per la struttura Body è necessario per poter inviare e ricevere istanze di Body tra processi MPI
    // Senza questo tipo di dato personalizzato, MPI non saprebbe come interpretare la struttura Body durante le comunicazioni
}

// Funzione per calcolare l'accelerazione di body1 dovuta alla forza gravitazionale esercitata da body2
void compute_acceleration(Body *body1, Body *body2, double acceleration[3]) {
    // Calcoliamo la distanza tra i due corpi
    // dx, dy, dz sono le differenze di posizione lungo ciascun asse
    // Il calcolo avviene sottraendo le coordinate di body1 da quelle di body2
    double dx = body2->position[0] - body1->position[0];
    double dy = body2->position[1] - body1->position[1];
    double dz = body2->position[2] - body1->position[2];

    // Calcoliamo la distanza euclidea tra i due corpi usando il teorema di Pitagora in 3D
    double distance = sqrt(dx*dx + dy*dy + dz*dz);
    // Calcoliamo la magnitudine della forza gravitazionale
    // La formula della forza gravitazionale è F = G * (m1 * m2) / r^2
    // Dove G è la costante gravitazionale, m1 e m2 sono le masse dei corpi, e r è la distanza tra di essi
    // Il significato fisico è che la forza diminuisce con il quadrato della distanza
    double magnitude = (G * body2->mass) / (distance * distance + 1e-10); // Aggiungiamo un piccolo termine per evitare la divisione per zero

    // Calcoliamo l'accelerazione come forza divisa per la massa di body1
    // L'accelerazione è un vettore che indica la direzione e la magnitudine della variazione della velocità
    acceleration[0] = magnitude * dx / (distance + 1e-10); // Componente x
    acceleration[1] = magnitude * dy / (distance + 1e-10); // Componente y
    acceleration[2] = magnitude * dz / (distance + 1e-10); // Componente z

    // La divisione per (distance + 1e-10) normalizza il vettore di direzione
    // In questo modo otteniamo un vettore unitario che indica la direzione della forza
    // L'accelerazione risultante viene quindi proporzionale alla forza gravitazionale esercitata da body2 su body1
}

// Facciamo un singolo step dell'approssimazione dell'integrale di runge kutta 4th order
void runge_kutta_step(Body *body, Body *all_bodies, int num_bodies) {
    // Definiamo le variabili per i coefficienti di Runge-Kutta
    // k1, k2, k3, k4 rappresentano le variazioni di velocità e posizione in ciascuna fase del metodo
    // Ogni k è un array di 3 elementi che rappresentano le componenti x, y, z
    // k1: variazione iniziale basata sull'accelerazione corrente
    // k2: variazione basata sull'accelerazione a metà passo
    // k3: variazione basata sull'accelerazione a metà passo
    // k4: variazione basata sull'accelerazione al passo successivo
    double k1v[3], k2v[3], k3v[3], k4v[3];
    double k1x[3], k2x[3], k3x[3], k4x[3];
    // Calcoliamo i 4 coefficienti di Runge-Kutta per ogni componente spaziale (x, y, z)
    // Il metodo RK4 approssima la soluzione dell'equazione differenziale mediante 4 stime incrementali
    for (int i = 0; i < 3; i++) {
        // K1: Prima stima - usa l'accelerazione corrente e la velocità corrente
        // Rappresenta la variazione lineare se l'accelerazione rimanesse costante per tutto il passo temporale
        k1v[i] = net_acceleration[i] * DT;  // Variazione di velocità nel primo quarto di passo
        k1x[i] = body->velocity[i] * DT;    // Variazione di posizione basata sulla velocità attuale
        
        // K2: Seconda stima - usa la stessa accelerazione ma con velocità modificata dalla prima stima
        // Simula cosa accadrebbe a metà del passo temporale utilizzando la velocità aggiornata da K1
        k2v[i] = net_acceleration[i] * DT;  // Variazione di velocità (stessa accelerazione)
        k2x[i] = (body->velocity[i] + 0.5 * k1v[i]) * DT;  // Posizione usando velocità a metà passo
        
        // K3: Terza stima - usa ancora la stessa accelerazione con la velocità corretta da K2
        // Fornisce una stima più accurata della variazione a metà passo usando il risultato di K2
        k3v[i] = net_acceleration[i] * DT;  // Variazione di velocità (stessa accelerazione)
        k3x[i] = (body->velocity[i] + 0.5 * k2v[i]) * DT;  // Posizione con velocità corretta da K2
        
        // K4: Quarta stima - usa l'accelerazione con la velocità al passo completo stimata da K3
        // Rappresenta la variazione che si avrebbe alla fine del passo temporale completo
        k4v[i] = net_acceleration[i] * DT;  // Variazione di velocità (stessa accelerazione)
        k4x[i] = (body->velocity[i] + k3v[i]) * DT;  // Posizione usando velocità al passo completo
    }
    
    // Applichiamo la formula finale di Runge-Kutta per aggiornare velocità e posizione
    // La formula combina le 4 stime con pesi specifici: 1/6 * (K1 + 2*K2 + 2*K3 + K4)
    // I coefficienti 1, 2, 2, 1 derivano dall'integrazione numerica e garantiscono accuratezza di ordine 4
    for (int i = 0; i < 3; i++) {
        // Aggiorniamo la velocità: peso maggiore dato alle stime intermedie (K2 e K3)
        // Questo riduce l'errore di troncamento e migliora la stabilità numerica
        body->velocity[i] += (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) / 6;
        
        // Aggiorniamo la posizione con la stessa formula pesata
        // Il risultato è una approssimazione molto più accurata rispetto all'integrazione di Eulero semplice
        body->position[i] += (k1x[i] + 2*k2x[i] + 2*k3x[i] + k4x[i]) / 6;
    }
}

// Aggiorniamo la posizione e la velocità di tutti i corpi nel sistema
// Questa funzione rappresenta il cuore dell'algoritmo parallelo: ogni processo calcola le nuove posizioni
// per il suo sottoinsieme di corpi (local_bodies) basandosi sulle posizioni di tutti i corpi (all_bodies)
void update_body_positions(Body *local_bodies, int local_count, Body *all_bodies, int rank) {
    // Iteriamo attraverso tutti i corpi assegnati a questo processo MPI
    // local_count rappresenta il numero di corpi di cui questo processo è responsabile
    for (int i = 0; i < local_count; i++) {
        // Inizializziamo l'accelerazione netta a zero per tutte e tre le componenti spaziali
        // net_acceleration accumula l'effetto gravitazionale di tutti gli altri corpi su local_bodies[i]
        double net_acceleration[3] = {0.0, 0.0, 0.0};
        
        // Calcoliamo la forza gravitazionale esercitata da ogni altro corpo nel sistema
        // È necessario considerare TUTTI i corpi (NUM_BODIES), non solo quelli locali
        // perché la forza gravitazionale è a lungo raggio e ogni corpo influenza tutti gli altri
        for (int j = 0; j < NUM_BODIES; j++) {
            // Controllo cruciale: evitiamo di calcolare l'auto-interazione (un corpo con se stesso)
            // Confrontiamo gli indirizzi dei puntatori per determinare se sono lo stesso oggetto
            // L'auto-interazione porterebbe a forze infinite o indefinite
            if (&local_bodies[i] != &all_bodies[j]) {
                // Calcoliamo l'accelerazione dovuta al corpo j-esimo
                double acc[3];
                compute_acceleration(&local_bodies[i], &all_bodies[j], acc);
                
                // Accumuliamo le componenti dell'accelerazione per il principio di sovrapposizione
                // In fisica, le forze si sommano vettorialmente
                net_acceleration[0] += acc[0];  // Componente x
                net_acceleration[1] += acc[1];  // Componente y
                net_acceleration[2] += acc[2];  // Componente z
            }
        }
        
        // Una volta calcolata l'accelerazione netta totale, applichiamo l'integrazione numerica
        // Utilizzando il metodo Runge-Kutta per aggiornare posizione e velocità
        // Questo passo risolve numericamente le equazioni differenziali del moto
        runge_kutta_step(&local_bodies[i], net_acceleration);
    }
}

// Generiamo un numero double casuale compreso tra min e max
double random_double(double min, double max) {
    // Usiamo la funzione rand() per generare un numero intero casuale
    // La funzione rand() restituisce un valore tra 0 e RAND_MAX
    double scale = rand() / (double) RAND_MAX; // Normalizziamo il valore tra 0 e 1
    // Scala il valore normalizzato al range desiderato [min, max]
    return min + scale * (max - min);
}

// Inizializziamo i corpi con masse, posizioni e velocità casuali
void initialize_random_bodies(Body *bodies, int num_bodies, unsigned int seed) {
    // Impostiamo il seme per il generatore di numeri casuali
    // Questo garantisce che ogni esecuzione con lo stesso seme produca la stessa sequenza di numeri casuali
    srand(seed);
    
    // Iteriamo attraverso tutti i corpi da inizializzare
    for (int i = 0; i < num_bodies; i++) {
        // Assegniamo una massa casuale tra 1e20 e 1e30 kg
        bodies[i].mass = random_double(1e20, 1e30);
        
        // Assegniamo una posizione casuale nello spazio tridimensionale
        // Le coordinate x, y, z sono comprese tra -1e13 e 1e13 metri
        bodies[i].position[0] = random_double(-1e13, 1e13); // x
        bodies[i].position[1] = random_double(-1e13, 1e13); // y
        bodies[i].position[2] = random_double(-1e13, 1e13); // z
        
        // Assegniamo una velocità casuale
        // Le componenti vx, vy, vz sono comprese tra -5e4 e 5e4 metri al secondo
        bodies[i].velocity[0] = random_double(-5e4, 5e4); // vx
        bodies[i].velocity[1] = random_double(-5e4, 5e4); // vy
        bodies[i].velocity[2] = random_double(-5e4, 5e4); // vz
    }
}

// Main function
int main(int argc, char **argv) {
    // Parsing dei parametri da linea di comando
    if (argc >= 3){
        NUM_BODIES = atoi(argv[1]);
        NUM_STEPS = atoi(argv[2]);
    }
    else {
        printf("Usage: %s <NUM_BODIES> <NUM_STEPS>\n", argv[0]);
        printf("Using default values: NUM_BODIES=%d, NUM_STEPS=%d\n", NUM_BODIES, NUM_STEPS);    }

    // Inizializziamo l'ambiente MPI
    // MPI_Init deve essere la prima funzione MPI chiamata nel programma
    // Questa funzione inizializza l'ambiente di esecuzione parallela e configura 
    // la comunicazione tra tutti i processi che partecipano alla simulazione
    int rank, size;    
    MPI_Init(&argc, &argv);
    
    // Otteniamo le informazioni fondamentali sui processi MPI
    // MPI_Comm_rank ottiene il rank (identificatore univoco) del processo corrente
    // Il rank è un numero intero che va da 0 a (size-1) e identifica univocamente ogni processo
    // Ad esempio: se abbiamo 4 processi, i rank saranno 0, 1, 2, 3
    // MPI_COMM_WORLD è il comunicatore globale che include tutti i processi avviati
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // MPI_Comm_size ottiene il numero totale di processi nel comunicatore
    // Questo valore determina quanti processi paralleli stanno eseguendo la simulazione
    // Se lanciamo il programma con "mpirun -np 4", allora size sarà 4
    // Ogni processo avrà lo stesso valore di size ma un rank diverso    
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Creiamo il tipo di dato MPI personalizzato per la struttura Body
    // Questo è necessario per poter trasmettere strutture Body complete tra processi MPI
    // La funzione create_body_datatype() definisce il layout della struttura in memoria per MPI
    MPI_Datatype body_type = create_body_datatype();
    
    // Allochiamo gli array per gestire la distribuzione dei corpi tra i processi
    // sendcounts: array che contiene il numero di corpi assegnati a ciascun processo
    // Ad esempio: se abbiamo 1000 corpi e 4 processi, sendcounts potrebbe essere [250, 250, 250, 250]
    // Qui stiamno solo allocando la memoria per questo array
    int *sendcounts = (int*) malloc(size * sizeof(int));
    
    // displs: array che contiene gli offset (spostamenti) per ogni processo nell'array globale
    // Indica dove inizia la porzione di dati di ciascun processo nell'array all_bodies
    // Continuando l'esempio: displs sarebbe [0, 250, 500, 750]
    // Se ci sono processi con carichi di lavoro diversi, displs aiuta a sapere dove iniziare a leggere/scrivere i dati
    // Ad esempio, se il processo 0 ha 300 corpi e il processo 1 ne ha 200, displs sarebbe [0, 300, 500, ...]
    int *displs = (int*) malloc(size * sizeof(int));
    
    // all_bodies: array globale che contiene TUTTI i corpi del sistema
    // Questo array è condiviso da tutti i processi e contiene le posizioni aggiornate di ogni corpo
    // È fondamentale per calcolare le forze gravitazionali, poiché ogni corpo influenza tutti gli altri
    Body *all_bodies = malloc(NUM_BODIES * sizeof(Body));
    
    // local_count: numero di corpi di cui questo specifico processo è responsabile
    // Sarà calcolato durante la distribuzione del carico di lavoro
    // Ogni processo avrà un valore diverso di local_count a seconda della sua porzione di lavoro    
    int local_count;

    // Creiamo il nome del file per salvare i risultati baseline della simulazione
    // Il file baseline contiene il tempo di esecuzione sequenziale (con 1 processo)
    // che viene usato come riferimento per calcolare speedup ed efficienza delle versioni parallele
    char baseline_filename[100];
    
    // snprintf costruisce il nome del file includendo i parametri della simulazione
    // Il formato è: "../results/baseline_<numero_corpi>_<numero_passi>.txt"
    // Ad esempio: "../results/baseline_1000_100.txt" per 1000 corpi e 100 passi temporali
    // Questo permette di avere baseline separati per ogni configurazione di test    
    snprintf(baseline_filename, sizeof(baseline_filename), "../results/baseline_%d_%d.txt", NUM_BODIES, NUM_STEPS);

    // Solo il processo master (rank 0) inizializza il sistema e calcola la distribuzione del carico
    // Questo evita duplicazioni e garantisce che tutti i processi partano dalla stessa configurazione iniziale
    if (rank == 0) {
        // Generiamo posizioni, velocità e masse casuali per tutti i corpi
        // Il seme fisso (10) garantisce riproducibilità tra diverse esecuzioni
        // Tutti i test useranno la stessa configurazione iniziale per confronti validi
        initialize_random_bodies(all_bodies, NUM_BODIES, 10);

        // Calcoliamo la distribuzione del carico di lavoro tra i processi
        // Dividiamo equamente i corpi, gestendo anche i casi in cui la divisione non è esatta
        int base_count = NUM_BODIES / size;  // Numero base di corpi per processo
        int remainder = NUM_BODIES % size;   // Corpi rimanenti da distribuire
        
        // Inizializziamo il primo displacement a 0 (il processo 0 inizia dall'indice 0)
        displs[0] = 0;
        
        // Calcoliamo sendcounts e displs per ogni processo
        for (int i = 0; i < size; i++) {
            // I primi 'remainder' processi ricevono un corpo extra per bilanciare il carico
            // Esempio: 1003 corpi con 4 processi -> [251, 251, 251, 250]
            sendcounts[i] = base_count + (i < remainder ? 1 : 0);
            
            // Calcoliamo dove inizia la porzione di ciascun processo nell'array all_bodies
            // displs[i] indica l'indice di partenza per il processo i
            if (i > 0) {
                // Questo calcolo accumula gli spostamenti
                // La logica è che il displacement del processo i è la somma del displacement del processo precedente
                // più il numero di corpi assegnati a quel processo precedente
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }   
    }

    // Distribuiamo le informazioni sul carico di lavoro a tutti i processi
    // MPI_Scatter invia a ogni processo il numero di corpi di cui sarà responsabile
    // sendcounts[i] viene inviato al processo i, che lo riceve in local_count
    // Solo il processo 0 ha l'array sendcounts completo, gli altri ricevono solo il loro valore
    // Esempio: se sendcounts = [251, 251, 251, 250], allora:
    // - processo 0 riceve local_count = 251
    // - processo 1 riceve local_count = 251 
    // - processo 2 riceve local_count = 251
    // - processo 3 riceve local_count = 250
    // i paramenti della funzione sono:
    // - sendbuf: array di invio (sendcounts)
    // - sendcount: numero di elementi inviati a ciascun processo (1 in questo caso)
    // - sendtype: tipo di dato degli elementi inviati (MPI_INT)
    // - recvbuf: variabile di ricezione (local_count)
    // - recvcount: numero di elementi ricevuti (1 in questo caso)
    // - recvtype: tipo di dato degli elementi ricevuti (MPI_INT)
    // - root: rank del processo che invia i dati (0 in questo caso)
    // - comm: comunicatore MPI (MPI_COMM_WORLD)
    // Automaticamente, ogni processo ottiene il proprio local_count tramite l'indice nel sendcounts
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Ogni processo alloca memoria per il proprio sottoinsieme di corpi
    // La dimensione dell'array locale dipende dal valore local_count ricevuto dallo scatter
    // Questo garantisce che ogni processo abbia esattamente la memoria necessaria
    // per gestire i corpi a lui assegnati, ottimizzando l'uso della memoria
    Body *local_bodies = malloc(local_count * sizeof(Body));

    // Iniziamo la misurazione del tempo di esecuzione parallela
    // MPI_Wtime() fornisce un timer ad alta precisione sincronizzato tra tutti i processi
    // Il timer viene avviato dopo tutta la fase di inizializzazione
    // per misurare solo il tempo effettivo di calcolo della simulazione    
    double start = MPI_Wtime();

    // Loop principale della simulazione: eseguiamo NUM_STEPS passi temporali
    // Ogni iterazione rappresenta un avanzamento nel tempo della simulazione fisica
    // Ad ogni passo, tutti i corpi si muovono secondo le forze gravitazionali calcolate
    for (int step = 0; step < NUM_STEPS; step++) {
        
        // FASE 1: Broadcast - Il processo 0 invia l'array completo all_bodies a tutti i processi
        // Questo è necessario perché ogni processo deve conoscere le posizioni di TUTTI i corpi
        // per calcolare correttamente le forze gravitazionali sui propri corpi locali
        // all_bodies contiene le posizioni aggiornate di tutti i corpi dopo l'iterazione precedente
        // I parametri sono:
        // - all_bodies: array di dati da inviare (solo nel processo 0)
        // - NUM_BODIES: numero di elementi nell'array
        // - body_type: tipo di dato MPI per la struttura Body
        // - 0: rank del processo che invia i dati (processo 0
        // - MPI_COMM_WORLD: comunicatore MPI
        MPI_Bcast(all_bodies, NUM_BODIES, body_type, 0, MPI_COMM_WORLD);
        
        // FASE 2: Scatterv - Distribuzione dei corpi ai processi per il calcolo parallelo
        // Ogni processo riceve la sua porzione specifica di corpi da aggiornare
        // MPI_Scatterv permette distribuzione non uniforme (sendcounts e displs differenti per processo)
        // I parametri sono:
        // - all_bodies: array sorgente (solo nel processo 0)
        // - sendcounts: numero di corpi inviati a ciascun processo
        // - displs: offset di partenza per ciascun processo nell'array all_bodies
        // - local_bodies: array di destinazione (locale a ogni processo)
        // - local_count: numero di corpi ricevuti da questo processo
        MPI_Scatterv(all_bodies, sendcounts, displs, body_type,
                     local_bodies, local_count, body_type,
                     0, MPI_COMM_WORLD);
        
        // FASE 3: Calcolo parallelo - Ogni processo aggiorna i suoi corpi locali
        // Questa è la fase computazionalmente intensiva della simulazione
        // Ogni processo calcola le nuove posizioni e velocità per i suoi corpi assegnati
        // utilizzando le informazioni di tutti i corpi ricevute dal broadcast
        update_body_positions(local_bodies, local_count, all_bodies, rank);

        // FASE 4: Gatherv - Raccolta dei risultati aggiornati nel processo 0
        // Ogni processo invia i suoi corpi aggiornati al processo 0
        // Il processo 0 ricompone l'array completo all_bodies con tutti gli aggiornamenti
        // Questo array aggiornato sarà usato nella prossima iterazione del loop
        // I parametri sono simili a Scatterv ma invertiti:
        // - local_bodies: array sorgente (locale a ogni processo)
        // - local_count: numero di corpi inviati da questo processo
        // - all_bodies: array di destinazione (solo nel processo 0)
        // - sendcounts: numero di corpi ricevuti da ciascun processo
        // - displs: offset di partenza per ciascun processo nell'array all_bodies
        // - body_type: tipo di dato MPI per la struttura Body
        MPI_Gatherv(local_bodies, local_count, body_type,
                    all_bodies, sendcounts, displs, body_type,
                    0, MPI_COMM_WORLD);
    }
      // Fine della misurazione del tempo: registriamo il tempo di completamento
    // La differenza (end - start) darà il tempo totale di esecuzione della simulazione
    double end = MPI_Wtime();

    // GESTIONE DEI RISULTATI E CALCOLO DELLE METRICHE DI PERFORMANCE
    // Solo il processo master (rank 0) gestisce l'output e i calcoli delle prestazioni
    // Questo evita output duplicati e garantisce un unico punto di controllo per i risultati
    if (rank == 0) {
        // Calcoliamo il tempo totale di esecuzione della simulazione
        double elapsed = end - start;
        
        // CASO 1: ESECUZIONE SEQUENZIALE (size == 1)
        // Quando usiamo un solo processo, salviamo il tempo come baseline di riferimento
        if (size == 1) {
            // Salviamo il tempo di esecuzione sequenziale nel file baseline
            // Questo tempo sarà usato come riferimento per calcolare speedup ed efficienza
            // delle versioni parallele con lo stesso numero di corpi e passi temporali
            FILE *file = fopen(baseline_filename, "w");
            if (file) {
                // Scriviamo il tempo con 6 decimali di precisione
                fprintf(file, "%.6f\n", elapsed);
                fclose(file);
            }
            
            // Output in formato CSV per l'esecuzione sequenziale
            // Formato: num_bodies,num_steps,processors,elapsed_time,speedup,efficiency
            // Per l'esecuzione sequenziale: speedup=0, efficiency=0 (valori convenzionali)
            printf("%d,%d,%d,%.6f,0.000000,0.000000\n", NUM_BODIES, NUM_STEPS, size, elapsed);
            
        } else {
            // CASO 2: ESECUZIONE PARALLELA (size > 1)
            // Calcoliamo speedup ed efficienza confrontando con il baseline sequenziale
            
            // Tentiamo di leggere il tempo baseline dal file salvato precedentemente
            FILE *file = fopen(baseline_filename, "r");
            double baseline = elapsed; // Default: usiamo il tempo corrente se non troviamo il baseline
            double speedup = 1.0;      // Speedup di default (nessun miglioramento)
            double efficiency = 1.0 / size; // Efficienza di default (pessimo caso)
            
            if (file) {
                // Leggiamo il tempo baseline dal file
                fscanf(file, "%lf", &baseline);
                fclose(file);
                
                // CALCOLO DELLO SPEEDUP
                // Speedup = T_sequenziale / T_parallelo
                // Indica quante volte è più veloce la versione parallela rispetto a quella sequenziale
                // Esempio: se T_seq = 100s e T_par = 25s, allora speedup = 4x
                // Speedup ideale = numero di processori, ma raramente raggiungibile a causa di overhead
                speedup = baseline / elapsed;
                
                // CALCOLO DELL'EFFICIENZA
                // Efficienza = Speedup / Numero_Processori
                // Misura quanto efficientemente stiamo utilizzando i processori disponibili
                // Varia da 0 a 1 (0% a 100%)
                // Esempio: speedup=3.2 con 4 processori → efficienza = 3.2/4 = 0.8 = 80%
                // Un'efficienza alta indica buon bilanciamento del carico e bassi overhead di comunicazione
                efficiency = speedup / size;
            }
            
            // Output in formato CSV per l'esecuzione parallela
            // Formato: num_bodies,num_steps,processors,elapsed_time,speedup,efficiency
            // Questi dati possono essere analizzati per valutare la scalabilità dell'algoritmo
            printf("%d,%d,%d,%.6f,%.6f,%.6f\n", NUM_BODIES, NUM_STEPS, size, elapsed, speedup, efficiency);
        }    }
    
    // SEZIONE DI PULIZIA DELLA MEMORIA E FINALIZZAZIONE MPI
    // È fondamentale rilasciare tutte le risorse allocate per prevenire memory leaks
    // e garantire una chiusura ordinata dell'ambiente parallelo
    
    // Deallocazione degli array principali - TUTTI I PROCESSI
    // Ogni processo deve liberare la memoria che ha allocato durante l'esecuzione
    
    // Liberiamo l'array globale che conteneva tutti i corpi del sistema
    // all_bodies è stato allocato da tutti i processi e contiene NUM_BODIES elementi
    // La dimensione in memoria è: NUM_BODIES * sizeof(Body) bytes
    free(all_bodies);
    
    // Liberiamo l'array locale che conteneva i corpi assegnati a questo processo
    // local_bodies ha dimensione diversa per ogni processo (local_count elementi)
    // Ogni processo ha allocato: local_count * sizeof(Body) bytes
    free(local_bodies);
    
    // Deallocazione degli array di distribuzione - SOLO IL PROCESSO MASTER
    // Solo il processo 0 ha allocato questi array per gestire la distribuzione del carico
    if (rank == 0) {
        // Liberiamo l'array sendcounts che conteneva il numero di corpi per processo
        // Dimensione: size * sizeof(int) bytes
        // Esempio: per 4 processi, liberiamo un array di 4 interi
        free(sendcounts);
        
        // Liberiamo l'array displs che conteneva gli offset per ogni processo
        // Dimensione: size * sizeof(int) bytes  
        // Questi array sono usati solo dal processo master nelle operazioni MPI collective
        free(displs);
    }
    
    // Deallocazione del tipo di dato MPI personalizzato
    // MPI_Type_free rilascia le risorse associate al tipo di dato body_type
    // È importante liberare i tipi MPI personalizzati per evitare memory leaks nell'ambiente MPI
    // Questo deve essere fatto da TUTTI i processi che hanno usato il tipo di dato
    MPI_Type_free(&body_type);
    
    // Finalizzazione dell'ambiente MPI
    // MPI_Finalize è l'ultima funzione MPI che deve essere chiamata nel programma
    // Questa funzione:
    // 1. Termina ordinatamente l'ambiente di esecuzione parallela
    // 2. Rilascia tutte le risorse MPI rimaste
    // 3. Sincronizza tutti i processi prima della terminazione
    // 4. Pulisce i comunicatori e le strutture dati interne di MPI
    // IMPORTANTE: dopo MPI_Finalize non è possibile chiamare altre funzioni MPI
    MPI_Finalize();
    
    // Terminazione normale del programma
    // return 0 indica successo al sistema operativo
    // Tutti i processi MPI devono raggiungere questo punto per una terminazione corretta
    return 0;
}
