// code for gibbs sampling for bayesian pmf considering univariate distributions

#include "../util/matrix.h"
#include "../util/random.h"
#include "../util/cmdline.h"
#include<map>
#include<ctime>
#include<vector>
#include<fstream>
#include<iostream>
#include<algorithm>
typedef unsigned int uint;

using namespace std;

struct sparse_entry {
    uint id;
    uint value;
};

struct data_store {
	uint user;
    uint movie;
    uint rating;
};

int main()
{
	// START - Initialising model parameter

    map< uint,uint > user_item_count;
    map< uint,uint > item_user_count;
    DVector<sparse_entry> train_user_item_pair;
    DVector<sparse_entry> test_user_item_pair;

	DVector<double> target;
	uint user_max=0,item_max=0;
    ifstream train("../../data/ra.train_sbpmf");
    uint num_ratings = 0;
    cout<<"hi1\n";
    while (!train.eof()) {

            uint movie_id,user_id;
            double rating;
            char whitespace;
			std::string line;
			std::getline(train, line);
			//const char *pline = line.c_str();
			if (sscanf(line.c_str(), "%u%c%u%c%lf", &user_id, &whitespace, &movie_id, &whitespace, &rating) >=5) {
                num_ratings++;
                if(user_max<user_id)
				{
					user_max = user_id;
				}
				if(item_max<movie_id)
				{
					item_max = movie_id;
				}

				if(user_item_count.find(user_id) == user_item_count.end())
				{
					user_item_count[user_id] = 1;
				}
				else
				{
					user_item_count[user_id]++;
				}

				if(item_user_count.find(movie_id) == item_user_count.end())
				{
					item_user_count[movie_id] = 1;
				}
				else
				{
					item_user_count[movie_id]++;
				}

			}
    }
    train.close();
    target.setSize(num_ratings);
    train_user_item_pair.setSize(num_ratings);

    train.open("../../data/ra.train_sbpmf");
    uint train_case = 0;
    while (!train.eof()) {

            uint movie_id,user_id;
            double rating;
            char whitespace;
			std::string line;
			std::getline(train, line);
			//const char *pline = line.c_str();
			if (sscanf(line.c_str(), "%u%c%u%c%lf", &user_id, &whitespace, &movie_id, &whitespace, &rating) >=5) {

                target(train_case) = rating;
                train_user_item_pair(train_case).id = user_id;
                train_user_item_pair(train_case).value = movie_id;
			}
			train_case++;
    }
    cout<<"11\n";
    train.close();
    ifstream test("../../data/ra.test_sbpmf");
    uint test_num_rows = 0;
    DVector<double> test_target;
    while (!test.eof()) {

            uint movie_id,user_id;
            double rating;
            char whitespace;
			std::string line;
			std::getline(test, line);
			//const char *pline = line.c_str();
			if (sscanf(line.c_str(), "%u%c%u%c%lf", &user_id, &whitespace, &movie_id, &whitespace, &rating) >=5)
            {
                test_num_rows++;
                if(user_max<user_id)
				{
					user_max = user_id;
				}
				if(item_max<movie_id)
				{
					item_max = movie_id;
				}
			}
    }
    test.close();
    cout<<"12\n";
    test_target.setSize(test_num_rows);
    test_user_item_pair.setSize(test_num_rows);
    DVector<double> sum;
    sum.setSize(test_num_rows);

    cout<<"13\n";
    test.open("../../data/ra.test_sbpmf");
    uint test_case = 0;
    while (!test.eof()) {

            uint movie_id,user_id;
            double rating;
            char whitespace;
			std::string line;
			std::getline(test, line);
			//const char *pline = line.c_str();
			if (sscanf(line.c_str(), "%u%c%u%c%lf", &user_id, &whitespace, &movie_id, &whitespace, &rating) >=5) {

                test_user_item_pair(test_case).id = user_id;
                test_user_item_pair(test_case).value = movie_id;
                test_target(test_case) = rating;
                sum(test_case) = 0.0;
			}
			test_case++;
    }
    test.close();
    cout<<"hi2\n";
    uint num_rows = num_ratings;
    uint num_users = user_max+1;
    uint num_items = item_max+1;

	// create R and R_t-----------------------------------------------------------------------------
    sparse_entry **R = new sparse_entry*[num_users];
    for(uint i=0; i<num_users; i++)
	{
		if(user_item_count.find(i) != user_item_count.end())
		{
			R[i] = new sparse_entry[user_item_count[i]];
		}
	}
	sparse_entry **R_t = new sparse_entry*[num_items];
    for(uint j=0; j<num_items; j++)
	{
		if(item_user_count.find(j) != item_user_count.end())
		{
			R_t[j] = new sparse_entry[item_user_count[j]];
		}
	}
	DVector<uint> user_index,item_index,user_num_item,item_num_user;
	user_index.setSize(num_users);
	item_index.setSize(num_items);
	user_num_item.setSize(num_users);
	item_num_user.setSize(num_items);
	user_index.init(0);
	item_index.init(0);
	user_num_item.init(0);
	item_num_user.init(0);

	for(uint i=0; i<num_users;i++)
	{
        if(user_item_count.find(i) != user_item_count.end())
        {
            user_num_item(i) = user_item_count[i];
        }
	}
	for(uint j=0; j<num_items;j++)
	{
        if(item_user_count.find(j) != item_user_count.end())
        {
            item_num_user(j) = item_user_count[i];
        }
	}

	train.open("../../data/ra.train_sbpmf");
	train_case = 0;
    while (!train.eof()) {

            uint movie_id,user_id;
            double rating;
            char whitespace;
			std::string line;
			std::getline(train, line);
			//const char *pline = line.c_str();
			if (sscanf(line.c_str(), "%u%c%u%c%lf", &user_id, &whitespace, &movie_id, &whitespace, &rating) >=5) {

                R[user_id][user_index(user_id)].id = train_case;
                R[user_id][user_index(user_id)].value = movie_id;
                R_t[movie_id][item_index(movie_id)].id = train_case;
                R_t[movie_id][item_index(movie_id)].value = user_id;
                user_index(user_id)++;
                item_index(movie_id)++;

			}
			train_case++;
    }
    train.close();

//---------------------------------------------------------------------------------------------------------------
    uint D = 20;
    cout<<"number rows ="<<num_rows<<"\n";
    cout<<"number of user ="<<num_users<<"\n";
    cout<<"number of items ="<<num_items<<"\n";

    double **U = new double*[num_users];
    for(uint i = 0; i < num_users; ++i) {
        U[i] = new double[D];
    }

    double **V = new double*[D];
    for(uint i = 0; i < D; ++i) {
        V[i] = new double[num_items];
    }

    // initialise U
    for( uint i=0; i<num_users; i++){
        for(uint j=0; j<D; j++){
            U[i][j] = 0.1*ran_gaussian(0.0,1.0);
        }
    }
    // initialise V
    for( uint j=0; j<D; j++){
        for(uint i=0; i<num_items; i++){
            V[j][i] = 0.1*ran_gaussian(0.0,1.0);
        }
    }

//---------------------------------------------------------------------------------------------------------

    DVectorDoubleVB sigma_u,mu_u,sigma_v,mu_v,mu_b_i,sigma_b_i,b_i,mu_b_j,sigma_b_j,b_j,temp_mu_u,temp_mu_v,temp_sigma_u,temp_sigma_v;

    sigma_u.setSize(D);
    mu_u.setSize(D);
    sigma_v.setSize(D);
    mu_v.setSize(D);
    temp_sigma_u.setSize(D);
    temp_mu_u.setSize(D);
    temp_sigma_v.setSize(D);
    temp_mu_v.setSize(D);

    mu_b_i.setSize(num_users);
    sigma_b_i.setSize(num_users);
    b_i.setSize(num_users);
    mu_b_j.setSize(num_items);
    sigma_b_j.setSize(num_items);
    b_j.setSize(num_items);

    sigma_u.init(0.02);
    sigma_v.init(0.02);
    //sigma_r.init(0.02);
    sigma_b_i.init(0.02);
    sigma_b_j.init(0.02);

    mu_u.init_normal(0.0,1.0);
    mu_v.init_normal(0.0,1.0);
    //mu_r.init_normal(0.0,1.0);
    mu_b_i.init_normal(0.0,1.0);
    mu_b_j.init_normal(0.0,1.0);

    b_i.init_normal(0.0,1.0);
    b_j.init_normal(0.0,1.0);
    //r.init_normal(0.0,1.0);

    double alpha_0 = 1.0;
    double alpha_1 = 1.0;
    double alpha_2 = 1.0;
    double alpha_3 = 1.0;
    double alpha_4 = 1.0;
    double alpha_5 = 1.0;
    double alpha_0_dash = 1.0;


    double beta_0 = 1.0;
    double beta_1 = 1.0;
    double beta_2 = 1.0;
    double beta_3 = 1.0;
    double beta_4 = 1.0;
    double beta_5 = 1.0;
    double beta_0_dash = 1.0;

    double mu_0 = 0.0;
    double mu_1 = 0.0;
    double mu_2 = 0.0;
    double mu_3 = 0.0;
    double mu_4 = 0.0;
    double mu_5 = 0.0;

    double sigma_0 = 1.0;
    double sigma_1 = 1.0;
    double sigma_2 = 1.0;
    double sigma_3 = 1.0;
    double sigma_4 = 1.0;
    double sigma_5 = 1.0;

    double mu_b_0 = 0.1*ran_gaussian(0.0,1.0);
    double sigma_b_0 = 0.02;
    double b_0 = 0.1*ran_gaussian(0.0,1.0);
    double alpha = ran_gamma(1.0,1.0);

//-------------------------------------------------------------------------------------------------------------------

    uint T = 100;
    uint burn_iter = 0;

    //double *E = new double[num_rows];
    cout<<"check1\n";
    DVector<double> E;
    E.setSize(num_rows);
    E.init(0.0);
    cout<<"check2\n";


	// End of initialization
    cout<<"hi3\n";
    for(uint iter = 0; iter<(T + burn_iter); iter++){

		time_t now=time(0);										// records time
		char * temp=ctime(&now);
		cout<<temp<<"\n";
		cout<<"iteration - "<<iter<<"\n";

        double E_sum_of_elements = 0.0;
        double E_sum_of_square_elements = 0.0;
		#pragma omp parallel for
        for(uint i=0; i<num_rows; i++)
        {
            //cout<<i<<"\t";
            uint user,item;
            user = train_user_item_pair(i).id;
            item = train_user_item_pair(i).value;
			//cout<<1<<"\t";
            double temp=0.0;
            for(uint k=0; k<D; k++)
            {
                temp+=U[user][k]*V[k][item];//*r(k);
            }
			//cout<<2<<"\n";
            E(i) = target(i) - (b_0 + b_i(user) + b_j(item) + temp);
            E_sum_of_elements += E(i);
            E_sum_of_square_elements += (E(i)*E(i));
        }

		temp_mu_u.init(0.0);
		temp_mu_v.init(0.0);
		temp_sigma_u.init(0.0);
		temp_sigma_v.init(0.0);
//--------------------------------------------------------------------------------------------------------------

	// Sample constants

        // for alpha

        double alpha_0_dash_star,beta_0_dash_star;

        alpha_0_dash_star = alpha_0_dash + num_rows;

        beta_0_dash_star = beta_0_dash + E_sum_of_square_elements;

        alpha = ran_gamma(alpha_0_dash_star,beta_0_dash_star);
        //cout<<"9\n";

        // for mu_b_0

        double sigma_0_star,mu_0_star;

        sigma_0_star = 1.0/(sigma_0 + sigma_b_0);
        mu_0_star = sigma_0_star * ((sigma_0*mu_0) + b_0*sigma_b_0);

        mu_b_0 = ran_gaussian(mu_0_star,sigma_0_star);
        //cout<<"3\n";

        // for sigma_b_0

        double alpha_0_star,beta_0_star;

        alpha_0_star = alpha_0 + 1;
        beta_0_star = beta_0 +  (0.5*(b_0 - mu_b_0)*(b_0 - mu_b_0));

        sigma_b_0 = ran_gamma(alpha_0_star,beta_0_star);
        //cout<<"7\n";

        // for b_0

        double sigma_b_0_star,mu_b_0_star;

        sigma_b_0_star = 1/(sigma_b_0 + alpha*num_rows);

        mu_b_0_star = sigma_b_0_star * (sigma_b_0*mu_b_0 + alpha*(E_sum_of_elements + num_rows*b_0));  //_________________

		double temp_b_0;
		temp_b_0=b_0;
        b_0 = ran_gaussian(mu_b_0_star,sigma_b_0_star);
		for(uint i=0; i<num_rows; i++)       // for calculation of summation terms in equation 1 and 2
		{
			E(i) += (temp_b_0-b_0);
		}
		//cout<<"6\n";


//---------------------------------------------------------------------------------------------------------------------------
	// sample parameters containing i
		#pragma omp parallel for
		for( uint i=0; i<num_users; i++)
		{
			// for mu_b_i

        	double sigma_4_star,mu_4_star;
			sigma_4_star = 1.0/(sigma_4 + sigma_b_i(i));
			mu_4_star = sigma_4_star * ((sigma_4*mu_4) + b_i(i)*sigma_b_i(i));

			mu_b_i(i) = ran_gaussian(mu_4_star,sigma_4_star);
	        //cout<<"1\n";

			// for sigma_b_i

        	double alpha_4_star,beta_4_star;
        	alpha_4_star = alpha_4 + 1;
            beta_4_star = beta_4 +  (0.5*(b_i(i) - mu_b_i(i))*(b_i(i) - mu_b_i(i)));

            sigma_b_i(i) = ran_gamma(alpha_4_star,beta_4_star);
        	//cout<<"4\n";

			// for b_i
        	double sigma_b_i_star,mu_b_i_star;

            sigma_b_i_star = 1/(sigma_b_i(i) + (alpha * user_item_count[i]));
            double temp=0.0;
            for( uint j=0; j<user_num_item(i); j++)
            {
                temp+= (E(R[i][j].id) + b_i(i));
            }
            mu_b_i_star = sigma_b_i_star * ((sigma_b_i(i)* mu_b_i(i))  + alpha*temp);

	    	double temp_b;
	    	temp_b=b_i(i);
            b_i(i) = ran_gaussian(mu_b_i_star,sigma_b_i_star);
	    	for(uint j=0; j<user_item_count[i]; j++)       // for calculation of summation terms in equation 1 and 2
            {
				E(R[i][j].id) += (temp_b-b_i(i));
	    	}
        	//cout<<"6\n";

			// for sampling U
        	for(uint k=0; k<D; k++)
			{
				temp_mu_u(k) += U[i][k];
				temp_sigma_u(k) += ((U[i][k] - mu_u(k)) * (U[i][k] - mu_u(k)));
				double temp=0.0,temp2=0.0,sigma_u_ik_star,mu_u_ik_star;
                for(uint j=0; j<user_item_count[i]; j++)       // for calculation of summation terms in equation 1 and 2
                {
                    temp += (V[k][R[i][j].value] * V[k][R[i][j].value]);// * r(k) * r(k)); //for equ 1
                    temp2 += (V[k][R[i][j].value] * (E(R[i][j].id) + V[k][R[i][j].value]*U[i][k]));//*r(k)) * r(k));
                }
                sigma_u_ik_star = 1/(sigma_u(k) + (alpha*temp));    //equation 12

                mu_u_ik_star = sigma_u_ik_star * (alpha*temp2 + sigma_u(k)*mu_u(k));  //equation 13

				double temp_u=U[i][k];
                U[i][k] = ran_gaussian(mu_u_ik_star,sigma_u_ik_star);
				for(uint j=0; j<user_item_count[i]; j++)       // for calculation of summation terms in equation 1 and 2
                {
					E(R[i][j].id) += V[k][R[i][j].value]*(temp_u-U[i][k]);//*r(k);
				}
			}

		}


//----------------------------------------------------------------------------------------------------------------------------

	// sampling parameters involving j
		#pragma omp parallel for
		for(uint j=0; j<num_items; j++)
		{
			// for mu_b_j

        	double sigma_5_star,mu_5_star;
            sigma_5_star = 1.0/(sigma_5 + sigma_b_j(j));
            mu_5_star = sigma_5_star * ((sigma_5*mu_5) + b_j(j)*sigma_b_j(j));

            mu_b_j(j) = ran_gaussian(mu_5_star,sigma_5_star);
        	//cout<<"2\n";

        	// for sigma_b_j

        	double alpha_5_star,beta_5_star;
        	alpha_5_star = alpha_5 + 1;
            beta_5_star = beta_5 +  (0.5*(b_j(j) - mu_b_j(j))*(b_j(j) - mu_b_j(j)));

            sigma_b_j(j) = ran_gamma(alpha_5_star,beta_5_star);
        	//cout<<"5\n";

        	// for b_j
        	double sigma_b_j_star,mu_b_j_star;
            sigma_b_j_star = 1/(sigma_b_j(j) + (alpha * item_user_count[j]));
            double temp=0.0;
            for( uint i=0; i<item_user_count[j]; i++)
            {
                temp+= (E(R_t[j][i].id) + b_j(j));
            }
            mu_b_j_star = sigma_b_j_star * ((sigma_b_j(j)* mu_b_j(j))  + alpha*temp);

	    	double temp_b;
	    	temp_b=b_j(j);
            b_j(j) = ran_gaussian(mu_b_j_star,sigma_b_j_star);
	    	for(uint i=0; i<item_user_count[j]; i++)       // for calculation of summation terms in equation 1 and 2
            {
				E(R_t[j][i].id) += (temp_b-b_j(j));
	    	}
			// for sampling V

	    	for(uint k=0; k<D; k++)
			{
				temp_mu_v(k) += V[k][j];
				temp_sigma_v(k) += (V[k][j] - mu_v(k)) * (V[k][j] - mu_v(k));
				double temp=0.0,temp2=0.0,sigma_v_jk_star,mu_v_jk_star;
                for(uint i=0; i<item_user_count[j]; i++)       // for calculation of summation terms in equation 3 and 4
                {
                    temp += (U[R_t[j][i].value][k] * U[R_t[j][i].value][k]);// * r(k) * r(k)); //for equ 3
                    temp2 += (U[R_t[j][i].value][k] * (E(R_t[j][i].id) + V[k][j]*U[R_t[j][i].value][k]));//*r(k)) * r(k)); //for equ 4
                }
                sigma_v_jk_star = 1/(sigma_v(k) + (alpha*temp));    //equation 3

                mu_v_jk_star = sigma_v_jk_star * (alpha*temp2 + sigma_v(k)*mu_v(k));  //equation 4

				double temp_v=V[k][j];
                V[k][j] = ran_gaussian(mu_v_jk_star,sigma_v_jk_star);

				for(uint i=0; i<item_user_count[j]; i++)       // for calculation of summation terms in equation 1 and 2
                {
					E(R_t[j][i].id) += U[R_t[j][i].value][k]*(temp_v-V[k][j]);//*r(k);
				}
			}
        }
        //cout<<"8\n";
//------------------------------------------------------------------------------------------------------------------------------
        // sampling of parameters with k dimensions
        #pragma omp parallel for
        for(uint k=0; k<D; k++){

            // for mu_u

            double sigma_2_star,mu_2_star;
            sigma_2_star = 1/(sigma_2 + sigma_u(k)*num_users);
            mu_2_star = sigma_2_star*(sigma_2*mu_2 + sigma_u(k)*temp_mu_u(k));

            mu_u(k) = ran_gaussian(mu_2_star,sigma_2_star);
            //cout<<"13\n";
            // for mu_v

            double sigma_1_star,mu_1_star;
            sigma_1_star = 1/(sigma_1 + sigma_v(k)*num_items);
			mu_1_star = sigma_1_star*(sigma_1*mu_1 + sigma_v(k)*temp_mu_v(k));

            mu_v(k) = ran_gaussian(mu_1_star,sigma_1_star);
            //cout<<"14\n";
            // for sigma_u

            double alpha_2_star,beta_2_star;

            alpha_2_star = alpha_2 + num_users;
            beta_2_star = beta_2 + (0.5)*temp_sigma_u(k);

            sigma_u(k) = ran_gamma(alpha_2_star,beta_2_star);
            //cout<<"15\n";
            // for sigma_v

            double alpha_1_star,beta_1_star;

            alpha_1_star = alpha_1 + num_items;
            beta_1_star = beta_1 + (0.5)*temp_sigma_v(k);

            sigma_v(k) = ran_gamma(alpha_1_star,beta_1_star);

            //cout<<"16\n";
        }
        //cout<<"19\n";
        if (iter>=0)
        {
            double diff_sqr_sum=0.0;
            double rmse;
            //cout<<"20\n";
            for(uint i=0; i<test_num_rows; i++)
            {
                //cout<<i<<"\t";
                uint user,item;
                double temp=0.0;
                user = test_user_item_pair(i).id;
                item = test_user_item_pair(i).value;
				temp=b_0 + b_i(user)+ b_j(item);
                for(uint k=0; k<D; k++)
                {
                    temp +=U[user][k]*V[k][item];//*r(k);
                }
                //cout<<"2\n";
				temp = std::min(5.0, temp);
				temp = std::max(0.5, temp);
                sum(i) += temp;
                diff_sqr_sum += (test_target(i) - ((double) sum(i)/(iter+1)))*(test_target(i) - ((double) sum(i)/(iter+1)));
            }
            //cout<<"21\n";
            rmse = sqrt(diff_sqr_sum/test_num_rows);

            std::cout<<"rmse is "<<rmse<<std::endl;
        }
    }
    cout<<"hi4\n";
    for(uint i = 0; i < num_users; ++i)
    {
        delete [] U[i];
    }
    delete [] U;

    for(uint i = 0; i < D; ++i)
    {
        delete [] V[i];
    }
    delete [] V;

    for(uint i = 0; i < num_users; ++i)
    {
    	if(user_item_count.find(i) != user_item_count.end())
		{
			delete[] R[i];
		}
    }
    delete [] R;

    for(uint j = 0; j < num_items; ++j)
    {
    	if(item_user_count.find(j) != item_user_count.end())
		{
			delete[] R_t[j];
		}
    }
    delete [] R_t;


return 0;
}
