import pytest

from aisecurity_rfds.common.report import pct, sci, table


def test_table_pads_every_column_to_its_widest_cell():
    out = table(["a", "long header"], [["x", 1], ["yy", 22]]).splitlines()
    assert out == [
        "| a  | long header |",
        "|----|-------------|",
        "| x  | 1           |",
        "| yy | 22          |",
    ]


def test_table_widths_come_from_the_cells_when_they_are_wider_than_the_header():
    assert table(["h"], [["wide cell"]]).splitlines()[0] == "| h         |"


def test_table_rejects_a_ragged_row():
    with pytest.raises(ValueError):
        table(["a", "b"], [["only one"]])


def test_table_with_no_rows_is_just_the_header_and_rule():
    assert table(["a", "b"], []).splitlines() == ["| a | b |", "|---|---|"]


@pytest.mark.parametrize(
    "x, expected",
    [(3.2e-5, "3.2e-5"), (0.0123, "1.2e-2"), (1234.0, "1.2e3"), (1.0, "1.0e0")],
)
def test_sci_strips_the_exponent_padding(x, expected):
    assert sci(x) == expected


def test_sci_digits_control_the_mantissa():
    assert sci(2.718281828, digits=3) == "2.718e0"


def test_pct_formats_a_fraction():
    assert pct(0.1234) == "12.3%"
    assert pct(1.0, digits=0) == "100%"
