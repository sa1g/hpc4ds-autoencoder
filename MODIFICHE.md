# Modifiche

## Codice

- Aggiunta una riga di startup in [main.cc] con dataset, mode, epoche, batch size, learning rate, world size e path della run.
- Sostituito il nome esperimento fisso `"prove"` con un nome generato automaticamente basato su dataset e modalità di esecuzione (`seq`, `omp`, `mpi`, `hybrid`).

- Aggiunta una stampa riassuntiva per epoca in [src/utils/worker.cc](/Users/silvanusbordignon/Documents/Università/4° anno/Primo semestre/High Performance Computing for Data Science/Project/hpc4ds-autoencoder/src/utils/worker.cc) con `train_loss`, `eval_loss`, `epoch_time_sec` e `samples_per_sec`.

## TensorBoard

- Aggiunto un controllo NaN/Inf in [src/utils/loops.cc](/Users/silvanusbordignon/Documents/Università/4° anno/Primo semestre/High Performance Computing for Data Science/Project/hpc4ds-autoencoder/src/utils/loops.cc): se la loss non è finita, viene stampato un errore e il loop termina subito.

- La cartella della run TensorBoard ora usa il formato `runs/<dataset>_<mode>_<timestamp>`.
- Il logger TensorBoard viene creato solo dal rank 0 in modalità MPI, così si evita di scrivere metriche duplicate su più run.
- Le metriche `train_loss` e `eval_loss` vengono mediate tra i rank MPI prima di essere stampate e salvate su TensorBoard.
- Aggiunti i nuovi scalar TensorBoard `epoch_time_sec` e `samples_per_sec`.
- Il `test_loss` finale viene ora salvato come metrica aggregata.
- Inserita una versione commentata per logging per-rank su TensorBoard, pronta da attivare se serve.
- Inseriti esempi commentati per logging dei pesi e dei gradienti.