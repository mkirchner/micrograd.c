#ifndef __MICROGRAD_VALUE_H__
#define __MICROGRAD_VALUE_H__

/*
 * In C, it's more convenient to allocate all values is a single array and
 * to manage relationships in a global adjacency (sparse) adjacency matrix.
 *
 */

#define allocate(x) malloc(x)

enum mgc_op {
  MGC_OP_NONE,
  MGC_OP_ADD,
  MGC_NOPS
};

struct mgc_val;
struct mgc_val {
    double value;
    double grad;
    struct mgc_val *p0, *p1;
    enum mgc_op op;
};
typedef struct mgc_val val;


struct mgc_val* mgc_value(double d);
void mgc_value_free(struct mgc_val *v);

#endif
