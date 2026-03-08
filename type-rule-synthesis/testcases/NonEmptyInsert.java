/*
annotation NonEmpty() ImmutableList
    :<==> "§subject§ != null && §subject§.length > 0"
    for "true";
*/

public final class NonEmptyInsert {

    public void insert(@NonEmpty ImmutableList l) {
        // False positive: the result of insert() should be NonEmpty
        // :: error: assignment.type.incompatible
        @NonEmpty ImmutableList l0 = l.insert(new Object());
    }

    public void remove(@PossiblyEmpty ImmutableList l) {
        // False negative: the argument of remove() should be NonEmpty, so there should be an error here.
        // Is your pipeline able to handle something like this too, or only false positives as above?
        l.remove(0);
    }

    public void removeTruePositive(@NonEmpty ImmutableList l) {
        // True positive: the result of remove() is PossiblyEmpty, so the error below should stay.
        // :: error: assignment.type.incompatible
        @NonEmpty ImmutableList l0 = l.remove(0);
    }
}

/*
Expected result:

---
v1.insert(v2) : @NonEmpty


v1 : @NonEmpty
---
v1.remove(v2) : @PossiblyEmpty
*/
