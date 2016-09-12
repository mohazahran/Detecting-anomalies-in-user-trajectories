# Detecting-anomalies-in-user-trajectories
The project is to model user's patterns and behavior, then using this model we can detect anomalies and outliers of the user's sequence of actions

Below is a couple of commands to run tribeflow on my data experiments 
=======================================================================================================
TotalLineCount For lastFM:  19150868

eclipse args==========================================================================================================================================
--num_iter 2000 --num_batches 20 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_topics 100 --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_userid-timestamp-artid-artname-traid-traname.dat --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/output.h5

using default small data
--num_iter 2000 --num_batches 20 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_topics 100 --trace_fpath /home/zahran/workspace/tribeFlow/zahranData/trace.dat --model_fpath /home/zahran/workspace/tribeFlow/zahranData/output.h5

using sampled data
--num_iter 2000 --num_batches 20 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_topics 100 --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_10k_sampledData --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/10k_sampled_output.h5


--num_iter 2000 --num_batches 20 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_topics 100 --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5


terminal running ==============================================================================================================================================

mpiexec -np 4 python /home/zahran/workspace/tribeFlow/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_userid-timestamp-artid-artname-traid-traname.dat --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/output.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20

mpiexec -np 4 python /home/zahran/workspace/tribeFlow/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20


old tribflow
mpiexec -np 4 python /home/zahran/workspace/tribeFlow/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20


new tribeflow
mpiexec -np 4 python /home/zahran/workspace/tribeFlow_new/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20

mpiexec -np 4 python /home/zahran/workspace/tribeFlow_new/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_B10_sampledData --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_B10_sampled_output.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20

mpiexec -np 4 python /home/zahran/workspace/tribeFlow_new/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_win5_pinterest.dat --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_win5_pinterest_model.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20

mpiexec -np 4 python /home/zahran/workspace/tribeFlow_new/main.py --trace_fpath /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_pinterest.dat --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_pinterest_model.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20

mpiexec -np 4 python /home/zahran/workspace/tribeFlow_new/main.py --trace_fpath /home/zahran/Desktop/shareFolder/PARSED_ALL_win10_pinterest --num_topics 100 --model_fpath /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_ALL_win10_pinterest_model.h5 --kernel eccdf --residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20




lastfm format
userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name

0 1 -d$'\t'


prepocessing=======================================================================================================================================================
Here, we are saying that column 1 are the timestamps, 0 is the user, and 2 are the
objects (artist ids). The delimiter *-d* is a tab. The time stamp format is
`'%Y-%m-%dT%H:%M:%SZ'`.

python /home/zahran/Desktop/tribeFlow/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_userid-timestamp-artid-artname-traid-traname.dat

using the sampled data
python /home/zahran/Desktop/tribeFlow/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/10k_sampledData 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_10k_sampledData

using the sampled data with memory param (B)
python /home/zahran/Desktop/tribeFlow/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/10k_sampledData 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' -m 2 > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_10k_sampledData

using the sampled data with memory param (B)
python /home/zahran/Desktop/tribeFlow/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' -m 2 > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData

using the sampled data with memory param (B) tribeflow_new
python /home/zahran/Desktop/tribeFlow/tribeFlow_new/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' -m 2 > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData



using the sampled data with memory param (B) tribeflow_new with very big B
python /home/zahran/Desktop/tribeFlow/tribeFlow_new/scripts/trace_converter.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData 1 0 2 -d$'\t' -f'%Y-%m-%dT%H:%M:%SZ' -m 10 > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_B10_sampledData

Zahran script
python /home/zahran/workspace/tribeFlow_new/scripts/trace_converter_zahran.py --original_trace /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData --tstamp_column 1 --hypernode_column 0 --obj_node_column 2 -m 10 -d \t -f'%Y-%m-%dT%H:%M:%SZ'  > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_B10_zahran_sampledData

python /home/zahran/workspace/tribeFlow_new/scripts/eventSequenceCreator_zahran.py --original_trace /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData --tstamp_column 1 --hypernode_column 0 --obj_node_column 2 -m 10 -d \t -f'%Y-%m-%dT%H:%M:%SZ'  > /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/SEQ_74123_B10_zahran_sampledData


eclipse args for preprocessing
--original_trace /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData --tstamp_column 1 --hypernode_column 0 --obj_node_column 2 -m 2 -d \t -f'%Y-%m-%dT%H:%M:%SZ' 
--original_trace /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampledData --tstamp_column 1 --hypernode_column 0 --obj_node_column 2 -m 10 -d \t -f'%Y-%m-%dT%H:%M:%SZ' 


prediction=======================================================================================
$ PYTHONPATH=. python scripts/mrr.py output.h5 rss.dat predictions.dat

python /home/zahran/Desktop/tribeFlow/tribeFlow_new/scripts/mrr.py /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5 /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PREDICTION_74123_sampledData

eclipse args:
/home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_sampled_output.h5 /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_sampledData /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PREDICTION_74123_sampledData

/home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/74123_B10_sampled_output.h5 /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PARSED_74123_B10_sampledData /home/zahran/Desktop/tribeFlow/zahranData/lastfm-dataset-1K/PREDICTION_74123_B10_sampledData

/home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_win5_pinterest_model.h5 /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PARSED_win5_pinterest.dat /home/zahran/Desktop/tribeFlow/zahranData/pinterest/PREDICTED_win5_pinterest.dat




