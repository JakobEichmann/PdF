public class IntervalExample {

    @Interval(min = 1, max = 3)
    int f;

    public void foo() {
        f = 3; // OK

        @Interval(min = 2, max = 4)
        int l = f + 1; // semantically OK, but checker without a rule rejects this
    }
}
