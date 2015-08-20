#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>

#define LEARN_RATE 1e-3

#define INPUT 4 /* N: Input nodes (given data) */
#define STORE 5 /* N + M: Storage nodes (encoded data) */
#define NEED 3 /* N - Delta: Minimumm number of nodes required to recover */

struct neuron;
struct synapse;

struct synapse {
	struct neuron *to;
	struct neuron *from;
	double weight;
};

struct neuron {
	struct synapse **next;
	struct synapse **prev;
	int next_size;
	int prev_size;
	int nodeid;
	int depth;
	double activation;
	double activationd;
	double sensitivity;
};

struct network {
	int input_bytes;
	int store_bytes;
	int need_bytes;
	int input_bits;
	int store_bits;
	int need_bits;
	int ncr_bytes;
	double input_bias;
	double store_bias;
	double *store_biases;
	struct neuron *inputs;
	struct neuron *stores;
	struct neuron **outputs;
};

double
sigmoid(double x)
{
        double expo, ret;

        expo = exp((double) -x);

        ret = 1 / (1 + expo);

        return ret;
}

double
ACT(double X)
{
        return sigmoid(X);

        if (sigmoid(X) < 0.5)
                return 0.0;
        else
                return 1.0;
//      return tanh(X);
}

double
ACTd(double X)
{
        double s = sigmoid(X);
        return s*(1-s);
//      double t = tanh(X);
//      return 1 - (t*t);
}

static int
facto(int i)
{
	if (!i) return 1;
	return i * facto(i-1);
}

static int
nCr(int n, int r)
{
	return facto(n) / (facto(r) * facto(n-r));
}

void
neuron_addnext(struct neuron *neuron, struct synapse *link)
{
	struct synapse **newnext;

	newnext = realloc(neuron->next, (neuron->next_size + 1)*sizeof(*newnext));
	if (!newnext)
		assert(!"Out of memory");
	newnext[neuron->next_size++] = link;
	neuron->next = newnext;
}

void
neuron_addprev(struct neuron *neuron, struct synapse *link)
{
	struct synapse **newprev;

	newprev = realloc(neuron->prev, (neuron->prev_size + 1)*sizeof(*newprev));
	if (!newprev)
		assert(!"Out of memory");
	newprev[neuron->prev_size++] = link;
	neuron->prev = newprev;
}

struct synapse *
neuron_link(struct neuron *from, struct neuron *to)
{
	struct synapse *link;

	link = calloc(1, sizeof(*link));
	if (!link)
		return NULL;

	link->weight = (double) (rand() - (RAND_MAX/2)) / RAND_MAX;
	link->from = from;
	link->to = to;

	neuron_addnext(from, link);
	neuron_addprev(to, link);

	printf("Linked: %d -> %d\n", from->nodeid, to->nodeid);

	return link;
}

struct network *
network_alloc(int input_bytes, int store_bytes, int need_bytes)
{
	struct network *network;
	struct neuron *inputs;
	struct neuron *stores;
	struct neuron **outputs;
	int ncr_bytes;

	ncr_bytes = nCr(store_bytes, need_bytes);

	network = calloc(1, sizeof(*network));
	if (!network)
		return NULL;

	network->input_bytes = input_bytes;
	network->store_bytes = store_bytes;
	network->need_bytes = need_bytes;
	network->ncr_bytes = ncr_bytes;
	network->input_bits = network->input_bytes * 8;
	network->store_bits = network->store_bytes * 8;
	network->need_bits = network->need_bytes * 8;

	inputs = calloc(network->input_bits, sizeof(*inputs));
	if (!inputs) {
		free(network);
		return NULL;
	}
	network->inputs = inputs;

	stores = calloc(network->store_bits, sizeof(*stores));
	if (!stores) {
		free(inputs);
		free(network);
		return NULL;
	}
	network->stores = stores;

	outputs = calloc(ncr_bytes, sizeof(*outputs));
	if (!outputs) {
		free(stores);
		free(inputs);
		free(network);
		return NULL;
	}

	for (int i = 0; i < ncr_bytes; i++) {
		outputs[i] = calloc(network->input_bits, sizeof(*outputs[i]));
		if (!outputs[i]) {
			for (int j = 0; j < i; j++)
				free(outputs[j]);
			free(outputs);
			free(stores);
			free(inputs);
			free(network);
			return NULL;
		}
	}
	network->outputs = outputs;

	return network;
}

typedef int output_cbk_t(int indices[], int seq, void *data);

static int
all_combos(int N, int R, int combo[], int depth, int max, int count, output_cbk_t cbk, void *data)
{
	int orig;

	if (cbk)
		cbk(combo, count, data);

	if (depth < 0) {
		return count;
	}

	orig = combo[depth];
	for (int i = combo[depth]+1; i < max; i++) {
		count++;
		combo[depth] = i;
		count = all_combos(N, R, combo, depth-1, i, count, cbk, data);
	}
	combo[depth] = orig;

	return count;
}

int
comb_enum(int N, int R, output_cbk_t cbk, void *data)
{
	int combo[R];
	int count = 0;

	for (int i = 0; i < R; i++)
		/* base combo */
		combo[i] = i;

	count = all_combos(N, R, combo, R-1, N, count, cbk, data);

	return count;
}


static int
output_link(int indices[], int seq, void *data)
{
	struct network *network = data;

	if (1) {
		printf("Combo(%d):", seq);
		for (int i = 0; i < network->need_bytes; i++)
			printf(" %d", indices[i]);
		printf("\n");
	}


	for (int i = 0; i < network->need_bytes; i++) {
		for (int j = 0; j < network->input_bits; j++) {
			for (int k = 0; k < 8; k++) {
				neuron_link(&network->stores[8*indices[i] + k], &network->outputs[seq][j]);
			}
		}
	}

	return 0;
}

void
network_init(struct network *network)
{
	int nodeid = 0;

        srand((unsigned int)time(NULL));
	network->input_bias = (double) (rand() - (RAND_MAX/2)) / RAND_MAX;
	network->store_bias = (double) (rand() - (RAND_MAX/2)) / RAND_MAX;
	network->store_biases = calloc(network->ncr_bytes, sizeof(*network->store_biases));
	for (int i = 0; i < network->ncr_bytes; i++)
		network->store_biases[i] = (double) (rand() - (RAND_MAX/2)) / RAND_MAX;

	for (int i = 0; i < network->input_bits; i++) {
		network->inputs[i].depth = 0;
		network->inputs[i].nodeid = ++nodeid;
	}

	for (int i = 0; i < network->store_bits; i++) {
		network->stores[i].depth = 1;
		network->stores[i].nodeid = ++nodeid;
	}

	for (int i = 0; i < network->ncr_bytes; i++) {
		for (int j = 0; j < network->input_bits; j++) {
			network->outputs[i][j].depth = 2;
			network->outputs[i][j].nodeid = ++nodeid;
		}
	}

	for (int i = 0; i < network->input_bits; i++)
		for (int j = 0; j < network->store_bits; j++)
			neuron_link(&network->inputs[i], &network->stores[j]);

	comb_enum(network->store_bytes, network->need_bytes, output_link, network);
}

struct network *
network_build(int input_bytes, int store_bytes, int need_bytes)
{
	struct network *network;

	network = network_alloc(input_bytes, store_bytes, need_bytes);
	if (!network)
		return NULL;

	network_init(network);
	return network;
}

double total_error = 0.0;

void
accept_input(struct network *network, FILE *fp)
{
/*
	for (int i = 0; i < network->input_bits; i++)
		if (((i/8) % 2) == 0)
			network->inputs[i].activation = rand() % 2;
		else
			network->inputs[i].activation = network->inputs[i-8].activation;
*/
	char bytes[network->input_bytes];
	memset(bytes, 0, network->input_bytes);

	fread(bytes, network->input_bytes, 1, fp);

	for (int i = 0; i < network->input_bits; i++)
		network->inputs[i].activation = (bytes[i / 8] >> (i % 8)) & 1;

	if (feof(fp)) {
		printf("Seeking back to 0. Total error=%f\n", total_error);
		total_error = 0.0;
		fseek(fp, 0, SEEK_SET);
	}
}

void
feedfwd(struct network *network)
{
	/* To store layer (hidden 1) */
	for (int i = 0; i < network->store_bits; i++) {
		struct neuron *to = &network->stores[i];
		double wTx = 0;
		for (int j = 0; j < to->prev_size; j++) {
			struct synapse *link = to->prev[j];
			struct neuron *from = link->from;

			wTx += (from->activation * link->weight);
		}

		to->activation = ACT(wTx + network->input_bias);
		to->activationd = ACTd(wTx + network->input_bias);
	}

	/* To each output layer (hidden 2) */
	for (int n = 0; n < network->ncr_bytes; n++) {
		for (int i = 0; i < network->input_bits; i++) {
			struct neuron *to = &network->outputs[n][i];
			double wTx = 0;
			for (int j = 0; j < to->prev_size; j++) {
				struct synapse *link = to->prev[j];
				struct neuron *from = link->from;

				wTx += (from->activation * link->weight);
			}

			to->activation = ACT(wTx + network->store_bias);
			to->activationd = ACTd(wTx + network->store_bias);
/*
			to->activation = ACT(wTx + network->store_biases[n]);
			to->activationd = ACTd(wTx + network->store_biases[n]);
*/

		}
	}
}

void
update_sensitivity(struct network *network)
{
	/* final layer, should match source */
	for (int n = 0; n < network->ncr_bytes; n++) {
		for (int i = 0; i < network->input_bits; i++) {
			struct neuron *target = &network->outputs[n][i];
			struct neuron *source = &network->inputs[i];

			target->sensitivity = 2 * (target->activation - source->activation);
		}
	}

	/* hidden layer */
	for (int i = 0; i < network->store_bits; i++) {
		struct neuron *from = &network->stores[i];
		double wTs = 0;

		for (int j = 0; j < from->next_size; j++) {
			struct synapse *link = from->next[j];
			struct neuron *to = link->to;

			wTs += (link->weight * to->sensitivity);
		}

		from->sensitivity = from->activationd * wTs;
	}

	/* input layer, redundant */
	for (int i = 0; i < network->input_bits; i++) {
		struct neuron *from = &network->inputs[i];
		double wTs = 0;

		for (int j = 0; j < from->next_size; j++) {
			struct synapse *link = from->next[j];
			struct neuron *to = link->to;

			wTs += (link->weight * to->sensitivity);
		}

		from->sensitivity = from->activationd * wTs;
	}
}

void
update_weights(struct network *network)
{
	/* input layer weights */
	for (int i = 0; i < network->input_bits; i++) {
		struct neuron *from = &network->inputs[i];
		for (int j = 0; j < from->next_size; j++) {
			struct synapse *link = from->next[j];
			struct neuron *to = link->to;

			link->weight -= LEARN_RATE * from->activation * to->sensitivity;
		}
	}
	/* input layer bias */
	for (int i = 0; i < network->store_bits; i++) {
		struct neuron *store = &network->stores[i];
		network->input_bias -= LEARN_RATE * store->sensitivity;
	}
	/* store layer weights */
	for (int i = 0; i < network->store_bits; i++) {
		struct neuron *from = &network->stores[i];
		for (int j = 0; j < from->next_size; j++) {
			struct synapse *link = from->next[j];
			struct neuron *to = link->to;

			link->weight -= LEARN_RATE * from->activation * to->sensitivity;
		}
	}
	/* store layer bias */
	for (int n = 0; n < network->ncr_bytes; n++) {
		for (int i = 0; i < network->input_bits; i++) {
			struct neuron *output = &network->outputs[n][i];
			network->store_bias -= LEARN_RATE * output->sensitivity;
/*
			network->store_biases[n] -= LEARN_RATE * output->sensitivity;
*/
		}
	}
}

void
backprop(struct network *network)
{
	update_sensitivity(network);
	update_weights(network);
}

void
printerr(struct network *network)
{
	double error = 0.0;

	for (int n = 0; n < network->ncr_bytes; n++) {
		for (int i = 0; i < network->input_bits; i++) {
			struct neuron *output = &network->outputs[n][i];
			struct neuron *input = &network->inputs[i];
			double d = (output->activation - input->activation);
			error += (d * d);
		}
	}

	total_error += error;

//	printf("Error: %f\n", error);
}

FILE *
data_prepare(const char *path)
{
	FILE *fp = fopen(path, "r");
	if (!fp)
		return NULL;
	return fp;
}

int
main(int argc, char *argv[])
{
	struct network *network;
	FILE *data;

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <input-data>\n", argv[0]);
		return 1;
	}

	network = network_build(INPUT, STORE, NEED);
	if (!network) {
		fprintf(stderr, "Out of memory\n");
		return 1;
	}

	data = data_prepare(argv[1]);
	if (!data) {
		fprintf(stderr, "%s: Unable to open file(%s)", argv[1], strerror(errno));
		return 1;
	}

	for (;;) {
		accept_input(network, data);
		feedfwd(network);
		printerr(network);
		backprop(network);
	}
}

