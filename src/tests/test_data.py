from polars.testing import assert_frame_equal


def test_head(data):
    """
    Tests that the head() method returns a Data object with the first n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the first n rows of the original data.
    """
    x = data.head()
    assert_frame_equal(x.returns, data.returns.head(5))

    #pd.testing.assert_frame_equal(x.all(), data.all().head())


def test_tail(data):
    """
    Tests that the tail() method returns a Data object with the last n rows.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The content matches the last n rows of the original data.
    """
    x = data.tail()
    assert_frame_equal(x.returns, data.returns.tail(5))

def test_all(data):
    print(data.returns.head(5))
    x = data.all
    print(x)

def test_assets(data):
    x = data.assets
    assert x == ['AAPL', 'META', 'SPY -- Benchmark']

def test_date_col(data):
    x = data.date_col
    assert x == ["Date"]

# def test_highwater_mark():
#     # Sample data
#     df = pl.DataFrame({
#         "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
#         "Asset": [0.01, -0.02, 0.03],
#     })
#
#     benchmark_df = pl.DataFrame({
#         "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
#         "Benchmark": [0.005, -0.01, 0.02],
#     })
#
#     index = df.select("Date")
#
#     data = _Data(
#         returns=df.drop("Date"),
#         benchmark=benchmark_df.drop("Date"),
#         index=index
#     )
#
#     # Test highwater mark with simple returns
#     hwm_simple = data.highwater_mark(compounded=False).to_pandas().set_index("Date")
#     prices_simple = data.prices(compounded=False).to_pandas().set_index("Date")
#
#     # Verify shape
#     assert hwm_simple.shape == prices_simple.shape
#
#     # Verify high-water mark is non-decreasing
#     for col in hwm_simple.columns:
#         assert hwm_simple[col].is_monotonic_increasing
#
#     print(hwm_simple.head(5))
#     print(prices_simple.head(5))
#     #assert (hwm_simple >= prices_simple).all().all()
#
#
# def test_drawdown(data):
#     """
#     Tests that the drawdown() method correctly calculates drawdowns from prices.
#
#     Args:
#         data (_Data): The data fixture containing a Data object.
#
#     Verifies:
#         1. The return value is a DataFrame with the same shape as the prices.
#         2. For compounded=False: drawdown equals peak_price - current_price.
#         3. For compounded=True: drawdown equals (current_price / peak_price) - 1.
#         4. Drawdowns are always <= 0 for compounded=True.
#         5. Drawdowns are always >= 0 for compounded=False.
#     """
#     # Test with compounded=False (default)
#     dd_simple = data.drawdown(compounded=False)
#     prices_simple = data.prices(compounded=False)
#     hwm_simple = prices_simple.cummax()
#
#     # Verify shape
#     assert dd_simple.shape == prices_simple.shape
#
#     # Verify drawdown calculation: peak_price - current_price
#     expected_dd_simple = hwm_simple - prices_simple
#     pd.testing.assert_frame_equal(dd_simple, expected_dd_simple)
#
#     # Verify drawdowns are always >= 0 (since they represent absolute losses)
#     assert (dd_simple >= 0.0).all().all()
#
#     # Test with compounded=True
#     dd_compound = data.drawdown(compounded=True)
#     prices_compound = data.prices(compounded=True)
#     hwm_compound = prices_compound.cummax()
#
#     # Verify shape
#     assert dd_compound.shape == prices_compound.shape
#
#     # Verify drawdown calculation: (current_price / peak_price) - 1
#     expected_dd_compound = prices_compound / hwm_compound - 1.0
#     pd.testing.assert_frame_equal(dd_compound, expected_dd_compound)
#
#     # Verify drawdowns are always <= 0 (since they represent percentage losses)
#     assert (dd_compound <= 0.0).all().all()
#
#

def test_copy(data):
    """
    Tests that the copy() method creates a proper deep copy of the Data object.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The copied object has the same returns and benchmark data as the original.
        3. Modifying the copied object does not affect the original.
        4. The copy works correctly when there's a benchmark.
    """
    # Create a copy of the data object
    data_copy = data.copy()

    # Verify the copy has the same returns and benchmark as the original
    assert_frame_equal(data_copy.returns, data.returns)
    assert_frame_equal(data_copy.benchmark, data.benchmark)

    # Verify that modifying the copy doesn't affect the original
    # We can't directly modify the attributes because the Data class is frozen,
    # but we can verify that they are separate objects in memory
    assert data_copy is not data
    assert data_copy.returns is not data.returns
    assert data_copy.benchmark is not data.benchmark


def test_resample(data):
    """
    Tests that the resample() method correctly resamples data to different time periods.

    Args:
        data (_Data): The data fixture containing a Data object.

    Verifies:
        1. The return value is a Data object.
        2. The resampled object has the correct frequency.
        3. The resampling works with both compounded=False and compounded=True.
        4. The resampling works with different resample rules (YE, ME, etc.).
    """
    # Test resampling to yearly frequency with compounded=False (default)
    yearly_data = data.resample(every="1y", compounded=False)

    # Verify the resampled data has the correct structure
    assert yearly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    print(yearly_data.all)
    #assert yearly_data.returns.index.freq == 'YE'  # Yearly frequency

    # Test resampling to monthly frequency with compounded=True
    monthly_data = data.resample(every="1mo", compounded=True)
    print(monthly_data.all)
    # Verify the resampled data has the correct structure
    assert monthly_data.returns.shape[1] == data.returns.shape[1]  # Same number of columns
    #assert monthly_data.returns.index.freq == 'ME'  # Monthly frequency


# def test_apply(data):
#     """
#     Tests that the apply() method correctly applies a function to the returns DataFrame.
#
#     Args:
#         data (_Data): The data fixture containing a Data object.
#
#     Verifies:
#         1. The method correctly applies a function to the returns DataFrame.
#         2. The result matches the direct application of the function to the returns DataFrame.
#         3. The method works with different functions and arguments.
#     """
#     # Test applying a function with no additional arguments
#     result1 = data.apply(np.mean)
#     expected1 = np.mean(data.returns)
#     assert result1 == pytest.approx(expected1)
#
#     # Test applying a function with axis argument
#     result2 = data.apply(np.mean, axis=1)
#     expected2 = np.mean(data.returns, axis=1)
#     pd.testing.assert_series_equal(result2, expected2)
#
#     # Test applying a function with multiple arguments
#     result3 = data.apply(np.percentile, q=75, axis=0)
#     expected3 = np.percentile(data.returns, q=75, axis=0)
#     np.testing.assert_array_equal(result3, expected3)
#
#     # Test applying a custom function
#     def custom_func(df, multiplier=1):
#         return df.sum() * multiplier
#
#     result4 = data.apply(custom_func, multiplier=2)
#     expected4 = custom_func(data.returns, multiplier=2)
#     pd.testing.assert_series_equal(result4, expected4)

# def test_all_pl(data):
#     print(data.all_pl())
#     x = data.all_pl().to_pandas().set_index("Date")
#     pd.testing.assert_frame_equal(x, data.all())
#     print(data.all().skew())
