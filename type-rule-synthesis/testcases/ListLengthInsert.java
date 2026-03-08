/*
annotation MinLength(int len) ImmutableList
    :<==> "§subject§ != null && §subject§.length > §len§"
    for "§len§ >= 0";
*/

public final class ListLengthInsert {

    public void insert(@MinLength("n") ImmutableList l) {
        // False positives: the result of insert() should be n+1
        // :: error: assignment.type.incompatible
        @MinLength("n") ImmutableList l0 = l.insert(new Object());
        // :: error: assignment.type.incompatible
        @MinLength("n+1") ImmutableList l1 = l.insert(new Object());

        // True positive
        // :: error: assignment.type.incompatible
        @MinLength("n+2") ImmutableList l2 = l.insert(new Object());
    }

    public void remove(@MinLength("1") ImmutableList l) {
        // False positive: the result of remove() should be length n-1
        @MinLength("0") ImmutableList l0 = l.remove(1);

        // False negative: the argument of remove() should be non-empty, so there should be an error here.
        // Is your pipeline able to handle something like this too, or only false positives as above?
        l0.remove(1);
    }

    public void removeTruePositive(@MinLength("1") ImmutableList l) {
        // True positive: the result of remove() is length n-1, the the error below should stay
        // :: error: assignment.type.incompatible
        @MinLength("1") ImmutableList l0 = l.remove(1);
    }
}

/*
Expected result:

v1 : MinLength(n)
---
v1.insert(v2) : MinLength(n+1)


n > 0
v1 : MinLength(n)
---
v1.remove(v2) : @MinLength(n-1)
*/
