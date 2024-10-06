# micrograd.c - micrograd in C
Autograd anywhere whith a C compiler

## Synopsis

Build a compute graph:

    /* y_hat = a*x + b */
    val *a = mgc_value(drand48());
    val *x = mgc_value(2.0);
    val *b = mgc_value(drand48());
    val *y_hat = mgc_add(mgc_mul(a, x), b);

Define a loss function:

    /* expected y */
    val *y = mgc_value(10.0);

    /* loss function: (y - y_hat)^2 */
    val *l = mgc_pow(mgc_add(y_hat, mgc_neg(y)), mgc_value(2));

Sort graph nodes by topological order:

    /* topological order over compute nodes */
    struct mgc_ref_vec sorted;
    mgc_ref_vec_init(&sorted);
    ptrdiff_t n = mgc_toposort(&sorted, l);

Gradient descent away:

    /* constant learning rate for simplicity */
    double h = 0.05;

    /* gradient descent over a, b */
    for (ptrdiff_t i = 0; i < 25; ++i) {
        /* forward pass */
        mgc_forward(&sorted);
        /* backward pass */
        mgc_zero_gradient(&sorted);
        l->grad = 1.0;
        mgc_backward(&sorted);
        /* parameter update */
        a->value -= h * a->grad;
        b->value -= h * b->grad;
        printf("loss=%g, a=%g, x=%g, b=%g\n", l->value, a->value, x->value,
               b->value);
    }

Which should produce something like:

    loss=78.8032, a=2.20278, x=2, b=1.15589
    loss=19.7008, a=3.09049, x=2, b=1.59975
    loss=4.9252, a=3.53434, x=2, b=1.82168
    loss=1.2313, a=3.75627, x=2, b=1.93264
    loss=0.307825, a=3.86723, x=2, b=1.98812
    loss=0.0769562, a=3.92272, x=2, b=2.01586
    loss=0.0192391, a=3.95046, x=2, b=2.02973
    loss=0.00480976, a=3.96433, x=2, b=2.03667
    loss=0.00120244, a=3.97126, x=2, b=2.04014
    [...]
    loss=1.11986e-12, a=3.9782, x=2, b=2.0436
    loss=2.79965e-13, a=3.9782, x=2, b=2.0436

