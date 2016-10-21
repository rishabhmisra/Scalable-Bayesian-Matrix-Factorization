// Author Rishabh Misra
// This file contains VB Online learning

#ifndef FM_LEARN_VB_ONLINE_SIMULTANEOUS_H_
#define FM_LEARN_VB_ONLINE_SIMULTANEOUS_H_

#include "fm_learn_vb_online.h"
#include "getRSS.c"
#include<ctime>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<sstream>
#include<iostream>
#include<algorithm>
#include<iterator>
#include<vector>
using namespace std;
class fm_learn_vb_online_simultaneous : public fm_learn_vb_online {
	protected:

		virtual void _learn(DataSubset& train, DataSubset& test)//,std::string train_filename,std::string test_filename)
		{

			uint num_complete_iter = 0;

			// make a collection of datasets that are predicted jointly
			int num_data = 2;
			int num_data_only_test = 1;

			DVector<DataSubset*> main_data_only_test(num_data_only_test);
			DVector<e_q_term*> main_cache_only_test(num_data_only_test);
			main_data_only_test(0) = &test;
			main_cache_only_test(0) = cache_test;

            uint num_batch = 10,tmp;
			total_cases = train.num_cases;
			DVector<uint> batch_size;
			batch_size.setSize(num_batch);
			tmp = ceil((double)(train.num_cases)/num_batch);
			size_except_last = tmp;
			batch_size.init(tmp);
			batch_size(num_batch-1) = (train.num_cases-(tmp*(num_batch-1)));
			
			uint* shuffle = new uint[train.num_cases];
            for(uint i=0;i<train.num_cases;i++)
            {
                shuffle[i] = i+1;
            }
            original_pos = new uint[train.num_cases];

			std::cout<<"check in fm_learn_vb_online_simultaneous"<<std::endl;
			long long int current,peak;
			current = getCurrentRSS();
			peak  = 12884901888;
			cout<<((double)current/peak)*100<<"%\n";
            ofstream batch[num_batch];
            for(uint i=0;i<num_batch;i++)
            {
                std::ostringstream sstream;
                sstream << "/home/hduser/avijit/rishabh/data/ra.train_libfm" <<"batch" << i+1;
                const char* file = (sstream.str()).c_str();
                batch[i].open(file, ios::out | ios::trunc);
            }
            uint index = 1;

            ifstream train_file("/home/hduser/avijit/rishabh/data/ra.train_libfm");                  //(train_filename.c_str());
            while (index<=train.num_cases) {

                std::string line;
                std::getline(train_file, line);
                uint group = ceil(((double)index/size_except_last));
                index++;
                batch[group-1] << line <<"\n";
            }
            train_file.close();

            for(uint i=0;i<num_batch;i++)
            {
                batch[i].close();
            }
			current = getCurrentRSS();
            
            cout<<((double)current/peak)*100<<"%\n";


            for(uint j = 1; j <= num_batch; j++)
            {
				{
					time_t now=time(0);                                     // records time
                    char * temp=ctime(&now);
                    std::cout<<"1 - "<<temp<<std::endl;
					std::ostringstream sstream;
					sstream << "/home/hduser/avijit/rishabh/data/ra.train_libfm" <<"batch"<< j;
					std::string file = sstream.str();
					//cout<<"inside3\n";
					current = getCurrentRSS();
                  
                    cout<<"before load "<<((double)current/peak)*100<<"%\n";

					DataSubset train2(0,true,true);
					train2.load(file,fm->num_attribute);

                    DVector<DataSubset*> main_data(1);       // to store train data
                    DVector<e_q_term*> main_cache(1);// to store intermediate term for calc. of Rij
		            main_data(0) = &train2;
		            main_cache(0) = cache;

                    now=time(0);                                     // records time
                    temp=ctime(&now);
                    std::cout<<"2 - "<<temp<<std::endl;
					current = getCurrentRSS();
            		cout<<"before eterms "<<((double)current/peak)*100<<"%\n";

					//for(uint i=0;i<10;i++)
                    predict_data_and_write_to_eterms(main_data, main_cache,j);

                    now=time(0);                                     // records time
                    temp=ctime(&now);
                    std::cout<<"3 - "<<temp<<std::endl;
					//for(uint i=0;i<10;i++)
					current = getCurrentRSS();
                    cout<<"before qterms "<<((double)current/peak)*100<<"%\n";

                    predict_t_and_write_to_qterms(&train2, cache_t,j);
					current = getCurrentRSS();
                    cout<<"before delete "<<((double)current/peak)*100<<"%\n";
			
					delete[] ((LargeSparseMatrixMemory<float>*) train2.data)->data.value[0].data;
					((LargeSparseMatrixMemory<float>*) train2.data)->data.setSize(0);
                    delete[] ((LargeSparseMatrixMemory<float>*) train2.data_t)->data.value[0].data;
					((LargeSparseMatrixMemory<float>*) train2.data_t)->data.setSize(0);
					delete ((LargeSparseMatrixMemory<float>*)train2.data);
					delete ((LargeSparseMatrixMemory<float>*)train2.data_t);
		
					current = getCurrentRSS();
                    cout<<((double)current/peak)*100<<"%\n";

				}
            }

			current = getCurrentRSS();
            cout<<"after loop "<<((double)current/peak)*100<<"%\n";
			// open file to store test rmse
			std::ofstream file_rmse;
			std::string file;
			std::stringstream convert; // stringstream used for the conversion

			convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

			std::string str = convert.str();
			file="test_rmse_" + str + "_vb_online";
			file_rmse.open(file.c_str());
			file_rmse.close();
			//clear free energy file
			std::ofstream myfile;
			file="free_energy_" + str + "_vb_online";
			myfile.open(file.c_str());
			myfile.close();
			//cout<<"check1\n";

//-------------------------------------------------------------------------------------------------------------------------

			//uint* shuffle = new uint[train.num_cases];
			//for(uint i=0;i<train.num_cases;i++)
			//{
			//	shuffle[i] = i+1;
			//}
			//original_pos = new uint[train.num_cases];
			//DVector<uint> batchIndex;
			//batchIndex.setSize(num_batch);
			//batchIndex.init(0);
//-------------------------------------------------------------------------------------------------------
			//cout<<"check2\n";

			current = getCurrentRSS();
            cout<<"before start "<<((double)current/peak)*100<<"%\n";
			for (uint k = num_complete_iter; k < num_iter; k++) {		// iterations over data
				time_t now=time(0);										// records time
				char * temp=ctime(&now);
				std::cout<<"4 - "<<temp<<std::endl;
				
				DVector<uint> batchIndex;
                batchIndex.setSize(num_batch);
                batchIndex.init(0);
//				current = getCurrentRSS();
//                cout<<"before batch loop "<<((double)current/peak)*100<<"%\n";
				double iteration_time = getusertime();
				clock_t iteration_time3 = clock();
				double iteration_time4 = getusertime4();
				nan_alpha=0; nan_sigma_0=0; nan_sigma_w=0; nan_sigma_v=0; nan_mu_0_dash=0; nan_sigma_0_dash=0; nan_mu_w_dash=0;
				nan_sigma_w_dash=0; nan_mu_v_dash=0; nan_sigma_v_dash=0; nan_sigma_w=0; nan_sigma_v=0;

//------------------------------------------------------------------------------------------------------------------------
				random_shuffle(shuffle,shuffle + train.num_cases);

				ofstream batch[num_batch];
				for(uint i=0;i<num_batch;i++)
				{
					std::ostringstream sstream;
					sstream << "/home/hduser/avijit/rishabh/data/ra.train_libfm" <<"batch" << i+1;
					const char* file = (sstream.str()).c_str();
					batch[i].open(file, ios::out | ios::trunc);
				}
				uint index = 0;
				//cout<<"inside1\n";
				ifstream train_file("/home/hduser/avijit/rishabh/data/ra.train_libfm");                  //(train_filename.c_str());
				while (index<train.num_cases) {

					std::string line;
					std::getline(train_file, line);
					tmp = shuffle[index];
					uint group = ceil(((double)tmp/size_except_last));
					original_pos[batchIndex(group-1) + (group-1)*size_except_last] = index;
					batchIndex(group-1)++;
					//cout<<tmp<<"\t"<<group<<"\n";
					batch[group-1] << line <<"\n";
					index++;
				}
				train_file.close();

				for(uint i=0;i<num_batch;i++)
				{
					batch[i].close();
				}
				//cout<<"inside2\n";
//				current = getCurrentRSS();
//                cout<<"loop starts here "<<((double)current/peak)*100<<"%\n";
				for(uint j = 1; j <= num_batch; j++)
				{
					time_t now=time(0);                                     // records time
                    char * temp=ctime(&now);
                    std::cout<<"5 - "<<temp<<std::endl;
					std::ostringstream sstream;
					sstream << "/home/hduser/avijit/rishabh/data/ra.train_libfm" <<"batch"<< j;
					std::string file = sstream.str();
					//cout<<"inside3\n";
//					current = getCurrentRSS();
//                    cout<<"before loading "<<((double)current/peak)*100<<"%\n";
					DataSubset train1(0,true,true); // no transpose data for sgd, sgda
					train1.load(file,fm->num_attribute);

//					current = getCurrentRSS();
//                    cout<<"after load "<<((double)current/peak)*100<<"%\n";
					now=time(0);                                     // records time^M
                    temp=ctime(&now);
                    std::cout<<"6 - "<<temp<<std::endl;
					//cout<<"inside4\n";
					if (task == TASK_REGRESSION) {
            	     // remove the target from each prediction, because: e(c) := \hat{y}(c) - target(c)
                        for (uint c = 0; c<train1.num_cases; c++) {
                            cache[original_pos[c + (j-1)*size_except_last]].e = train1.target(c) - cache[original_pos[c + (j-1)*size_except_last]].e;
                        }

                    } else if (task == TASK_CLASSIFICATION) {
	                 // for Classification: remove from e not the target but a sampled value from a truncated normal^M
    	             // for initializing, they are not sampled but initialized with meaningful values:^M
        	         // -1 for the negative class and +1 for the positive class (actually these are the values that are already in the target an    d thus, we can do the same as for regression; but note that other initialization strategies would need other techniques here:
                        for (uint c = 0; c<train1.num_cases; c++) {
                            cache[original_pos[c + (j-1)*size_except_last]].e = train1.target(c) - cache[original_pos[c + (j-1)*size_except_last]].e;
                        }

                    } else {
    		             throw "unknown task";
	        	     }
					//cout<<"inside5\n";
//					current = getCurrentRSS();
 //                  cout<<"before update "<<((double)current/peak)*100<<"%\n";
                    update_all(train1,j);
//					current = getCurrentRSS();
 //                   cout<<"after update "<<((double)current/peak)*100<<"%\n";
					//cout<<"hi1\n";

					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value[0].data;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value[0].data;
					delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value;

//					delete[] ((LargeSparseMatrixMemory<float>*) train1.data)->data.value[0].data;
  //                  ((LargeSparseMatrixMemory<float>*) train1.data)->data.setSize(0);
    //                delete[] ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.value[0].data;
      //              ((LargeSparseMatrixMemory<float>*) train1.data_t)->data.setSize(0);
        //            delete ((LargeSparseMatrixMemory<float>*)train1.data);
          //          delete ((LargeSparseMatrixMemory<float>*)train1.data_t);
                }
//				current = getCurrentRSS();
 //               cout<<"loop end "<<((double)current/peak)*100<<"%\n";
				neww = tau*neww;
//				cout<<"check3\n";
				now=time(0);                                     // records time^M
                temp=ctime(&now);
                std::cout<<"7 - "<<temp<<std::endl;
//---------------------------------------------------------------------------------------------------------------

				//std::cout<<"check in fm_learn_vb_sim"<<std::endl;
				if ((nan_alpha > 0) || (inf_alpha > 0)) {
					std::cout << "#nans in alpha:\t" << nan_alpha << "\t#inf_in_alpha:\t" << inf_alpha << std::endl;
				}
				if ((nan_sigma_0 > 0) || (inf_sigma_0 > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_0<< "\t#inf_in_alpha:\t" << inf_sigma_0 << std::endl;
				}
				if ((nan_sigma_w > 0) || (inf_sigma_w > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_w << "\t#inf_in_alpha:\t" << inf_sigma_w << std::endl;
				}
				if ((nan_sigma_v > 0) || (inf_sigma_v > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_v << "\t#inf_in_alpha:\t" << inf_sigma_v << std::endl;
				}
				if ((nan_mu_0_dash > 0) || (inf_mu_0_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_0_dash << "\t#inf_in_alpha:\t" << inf_mu_0_dash << std::endl;
				}
				if ((nan_sigma_0_dash > 0) || (inf_sigma_0_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_0_dash << "\t#inf_in_alpha:\t" << inf_sigma_0_dash << std::endl;
				}
				if ((nan_mu_w_dash > 0) || (inf_mu_w_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_w_dash << "\t#inf_in_alpha:\t" << inf_mu_w_dash << std::endl;
				}
				if ((nan_sigma_w_dash > 0) || (inf_sigma_w_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_w_dash << "\t#inf_in_alpha:\t" << inf_sigma_w_dash << std::endl;
				}
				if ((nan_mu_v_dash > 0) || (inf_mu_v_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_mu_v_dash << "\t#inf_in_alpha:\t" << inf_mu_v_dash << std::endl;
				}
				if ((nan_sigma_v_dash > 0) || (inf_sigma_v_dash > 0)) {
					std::cout << "#nans in alpha:\t" << nan_sigma_v_dash << "\t#inf_in_alpha:\t" << inf_sigma_v_dash << std::endl;
				}



				// predict test and train
				//std::cout<<"Updating error term"<<std::endl;
				//predict_data_and_write_to_eterms(main_data, main_cache);
				predict_data_and_write_to_eterms(main_data_only_test, main_cache_only_test);
				// (prediction of train is not necessary but it increases numerical stability)

//				cout<<"after test_pred\n";
				std::ofstream file_rmse;
				std::string file;
				std::stringstream convert; // stringstream used for the conversion

				convert << fm->k0<<fm->k1<<fm->num_factor;//add the value of Number to the characters in the stream

				std::string str = convert.str();
				file="test_rmse_" + str + "_vb_online";
				file_rmse.open(file.c_str(), std::ios_base::app);

				double acc_train = 0.0;
				double rmse_train = 0.0;
				if (task == TASK_REGRESSION) {
					// evaluate test and store it
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						//std::cout<<p<<std::endl;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						//std::cout<<p<<std::endl;
						pred_this(c) = p;
					}

					// Evaluate the training dataset and update the e-terms
					/*for (uint c = 0; c < train.num_cases; c++) {
						double p = cache[c].e;
						p = std::min(max_target, p);
						p = std::max(min_target, p);
						//double err = train.target(c) - p;
						//rmse_train += err*err;
						rmse_train += p*p;
						//cache[c].e = train.target(c) - cache[c].e;
					}
					rmse_train = std::sqrt(rmse_train/train.num_cases);*/

				} else if (task == TASK_CLASSIFICATION) {
					// evaluate test and store it
					for (uint c = 0; c < test.num_cases; c++) {
						double p = cache_test[c].e;
						p = cdf_gaussian(p);
						pred_this(c) = p;
					}

					// Evaluate the training dataset and update the e-terms
					/*uint _acc_train = 0;
					for (uint c = 0; c < train.num_cases; c++) {
						double p = cache[c].e;
						p = cdf_gaussian(p);
						if (((p >= 0.5) && (train.target(c) > 0.0)) || ((p < 0.5) && (train.target(c) < 0.0))) {
							_acc_train++;
						}

						double sampled_target;
						if (train.target(c) >= 0.0) {
							{
								// the target is the expected value of the truncated normal
								double mu = cache[c].e;
								double phi_minus_mu = exp(-mu*mu/2.0) / sqrt(3.141*2);
								double Phi_minus_mu = cdf_gaussian(-mu);
								sampled_target = mu + phi_minus_mu / (1-Phi_minus_mu);
							}
						} else {
							{
								// the target is the expected value of the truncated normal
								double mu = cache[c].e;
								double phi_minus_mu = exp(-mu*mu/2.0) / sqrt(3.141*2);
								double Phi_minus_mu = cdf_gaussian(-mu);
								sampled_target = mu - phi_minus_mu / Phi_minus_mu;
							}
						}
						cache[c].e = sampled_target - cache[c].e ;
					}
					acc_train = (double) _acc_train / train.num_cases;*/

				} else {
					throw "unknown task";
				}

				iteration_time = (getusertime() - iteration_time);
				iteration_time3 = clock() - iteration_time3;
				iteration_time4 = (getusertime4() - iteration_time4);
				if (log != NULL) {
					log->log("time_learn", iteration_time);
					log->log("time_learn2", (double)iteration_time3 / CLOCKS_PER_SEC);
					log->log("time_learn4", (double)iteration_time4);
				}


				// Evaluate the test data sets
				if (task == TASK_REGRESSION) {
					double rmse_test_this, mae_test_this;
					_evaluate(pred_this, test.target, 1.0, rmse_test_this, mae_test_this, num_eval_cases);
					file_rmse<<rmse_test_this<<"\n";
					std::cout << "#Iter=" << std::setw(3) << k << "\tTest=" << rmse_test_this << std::endl;

					if (log != NULL) {
						log->log("rmse_mcmc_this", rmse_test_this);

						if (num_eval_cases < test.target.dim) {
							double rmse_test2_this, mae_test2_this;//, rmse_test2_all_but5, mae_test2_all_but5;
							 _evaluate(pred_this, test.target, 1.0, rmse_test2_this, mae_test2_this, num_eval_cases, test.target.dim);
							//log->log("rmse_mcmc_test2_this", rmse_test2_this);
							//log->log("rmse_mcmc_test2_all", rmse_test2_all);
						}
						log->newLine();
					}
				} else if (task == TASK_CLASSIFICATION) {
					double acc_test_this, ll_test_this;
					 _evaluate_class(pred_this, test.target, 1.0, acc_test_this, ll_test_this, num_eval_cases);

					std::cout << "#Iter=" << std::setw(3) << k << "\tTest=" << acc_test_this << "\tTest(ll)=" << ll_test_this << std::endl;

					if (log != NULL) {
						log->log("acc_mcmc_this", acc_test_this);
						log->log("ll_mcmc_this", ll_test_this);

						if (num_eval_cases < test.target.dim) {
							double acc_test2_this,ll_test2_this;
							 _evaluate_class(pred_this, test.target, 1.0, acc_test2_this, ll_test2_this, num_eval_cases, test.target.dim);
							//log->log("acc_mcmc_test2_this", acc_test2_this);
							//log->log("acc_mcmc_test2_all", acc_test2_all);
						}
						log->newLine();
					}

				} else {
					throw "unknown task";
				}
				file_rmse.close();
			}
			delete[] shuffle;
			delete[] original_pos;
		}

		void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint from_case, uint to_case) {
			assert(pred.dim == target.dim);
			double _rmse = 0;
			double _mae = 0;
			uint num_cases = 0;
			for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
				double p = pred(c) * normalizer;
				p = std::min(max_target, p);
				p = std::max(min_target, p);
				double err = p - target(c);
				_rmse += err*err;
				_mae += std::abs((double)err);
				num_cases++;
			}

			rmse = std::sqrt(_rmse/num_cases);
			mae = _mae/num_cases;

		}

		void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint from_case, uint to_case) {
			double _loglikelihood = 0.0;
			uint _accuracy = 0;
			uint num_cases = 0;
			for (uint c = std::max((uint) 0, from_case); c < std::min((uint)pred.dim, to_case); c++) {
				double p = pred(c) * normalizer;
				if (((p >= 0.5) && (target(c) > 0.0)) || ((p < 0.5) && (target(c) < 0.0))) {
					_accuracy++;
				}
				double m = (target(c)+1.0)*0.5;
				double pll = p;
				if (pll > 0.99) { pll = 0.99; }
				if (pll < 0.01) { pll = 0.01; }
				_loglikelihood -= m*log10(pll) + (1-m)*log10(1-pll);
				num_cases++;
			}
			loglikelihood = _loglikelihood/num_cases;
			accuracy = (double) _accuracy / num_cases;
		}


		void _evaluate(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& rmse, double& mae, uint& num_eval_cases) {
			_evaluate(pred, target, normalizer, rmse, mae, 0, num_eval_cases);
		}

		void _evaluate_class(DVector<double>& pred, DVector<DATA_FLOAT>& target, double normalizer, double& accuracy, double& loglikelihood, uint& num_eval_cases) {
			_evaluate_class(pred, target, normalizer, accuracy, loglikelihood, 0, num_eval_cases);
		}
};

#endif /*FM_LEARN_MCMC_SIMULTANEOUS_H_*/
