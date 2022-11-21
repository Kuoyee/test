from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql import HiveContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def Spark_connection_hive():
    spark_session = SparkSession.builder.master("local").appName("rent_analyse").getOrCreate()
    hive_context = HiveContext(spark_session)

    # 生成查询的SQL语句，这个跟hive的查询语句一样，所以也可以加where等条件语句
    hive_database = "xmzf"
    hive_table = "rent"
    hive_read = "select * from  {}.{}".format(hive_database, hive_table)

    # 通过SQL语句在hive中查询的数据直接是dataframe的形式
    df = hive_context.sql(hive_read)
    return df

def Spark_connection(filename):
    # 程序主入口
    spark = SparkSession.builder.master("local").appName("rent_analyse").getOrCreate()
    df = spark.read.csv(filename, header=True,encoding='gbk')
    return df
def Data_preprocessing(df):
    print("开始数据预处理")
    df = df.withColumn("IDhouse", df.IDhouse.cast(IntegerType()))
    df = df.withColumn("District_id", df.District_id.cast(IntegerType()))
    df = df.withColumn("Price", df.Price.cast(IntegerType()))
    df = df.withColumn("Rentmode", df.Rentmode.cast(IntegerType()))
    df = df.withColumn("Room", df.Room.cast(IntegerType()))
    df = df.withColumn("Hall", df.Hall.cast(IntegerType()))
    df = df.withColumn("Bathroom", df.Bathroom.cast(IntegerType()))
    df = df.withColumn("Area", df.Area.cast(FloatType()))
    df = df.withColumn("Decoration", df.Decoration.cast(IntegerType()))
    df = df.withColumn("Orientation", df.Orientation.cast(IntegerType()))
    df = df.withColumn("Maintain", df.Maintain.cast(IntegerType()))
    df = df.withColumn("Floor", df.Floor.cast(IntegerType()))
    df = df.withColumn("Elevator", df.Elevator.cast(IntegerType()))
    df = df.withColumn("Parking", df.Parking.cast(IntegerType()))
    df = df.withColumn("Gas", df.Gas.cast(IntegerType()))
    df = df.withColumn("Washingmachine", df.Washingmachine.cast(IntegerType()))
    df = df.withColumn("Airconditioner", df.Airconditioner.cast(IntegerType()))
    df = df.withColumn("Wardrobe", df.Wardrobe.cast(IntegerType()))
    df = df.withColumn("TV", df.TV.cast(IntegerType()))
    df = df.withColumn("Refrigerator", df.Refrigerator.cast(IntegerType()))
    df = df.withColumn("Waterheater", df.Waterheater.cast(IntegerType()))
    df = df.withColumn("Bed", df.Bed.cast(IntegerType()))
    df = df.withColumn("Heating", df.Heating.cast(IntegerType()))
    df = df.withColumn("Broadband", df.Broadband.cast(IntegerType()))
    df = df.withColumn("Naturalgas", df.Naturalgas.cast(IntegerType()))
    return df

def spark_analyse_decoration(df):
    mean_price_list = [1.2 for i in range(2)]
    mean_area_list=[1.2 for i in range(2)]
    mean_list = [1.2 for i in range(2)]
    #计算各装修级别的平均值
    mean_price_list[0] = df.filter(df.Decoration == 0).agg({"Price": "mean"}).first()['avg(Price)']
    mean_price_list[1] = df.filter(df.Decoration == 1).agg({"Price": "mean"}).first()['avg(Price)']

    mean_area_list[0] = df.filter(df.Decoration == 0).agg({"Area": "mean"}).first()['avg(Area)']
    mean_area_list[1] = df.filter(df.Decoration == 1).agg({"Area": "mean"}).first()['avg(Area)']

    mean_list[0] = mean_price_list[0] / mean_area_list[0]
    mean_list[1] = mean_price_list[1] / mean_area_list[1]

    return mean_list
def spark_analyse_area(df):
    max_list = [0 for i in range(6)]
    mean_list = [1.2 for i in range(6)]
    min_list = [0 for i in range(6)]
    mid_list = [0 for i in range(6)]

    mean_list[0] = df.filter(df.District == "海沧").agg({"Area": "mean"}).first()['avg(Area)']
    mean_list[1] = df.filter(df.District == "湖里").agg({"Area": "mean"}).first()['avg(Area)']
    mean_list[2] = df.filter(df.District == "集美").agg({"Area": "mean"}).first()['avg(Area)']
    mean_list[3] = df.filter(df.District == "思明").agg({"Area": "mean"}).first()['avg(Area)']
    mean_list[4] = df.filter(df.District == "翔安").agg({"Area": "mean"}).first()['avg(Area)']
    mean_list[5] = df.filter(df.District == "同安").agg({"Area": "mean"}).first()['avg(Area)']

    min_list[0] = df.filter(df.District == "海沧").agg({"Area": "min"}).first()['min(Area)']
    min_list[1] = df.filter(df.District == "湖里").agg({"Area": "min"}).first()['min(Area)']
    min_list[2] = df.filter(df.District == "集美").agg({"Area": "min"}).first()['min(Area)']
    min_list[3] = df.filter(df.District == "思明").agg({"Area": "min"}).first()['min(Area)']
    min_list[4] = df.filter(df.District == "翔安").agg({"Area": "min"}).first()['min(Area)']
    min_list[5] = df.filter(df.District == "同安").agg({"Area": "min"}).first()['min(Area)']

    max_list[0] = df.filter(df.District == "海沧").agg({"Area": "max"}).first()['max(Area)']
    max_list[1] = df.filter(df.District == "湖里").agg({"Area": "max"}).first()['max(Area)']
    max_list[2] = df.filter(df.District == "集美").agg({"Area": "max"}).first()['max(Area)']
    max_list[3] = df.filter(df.District == "思明").agg({"Area": "max"}).first()['max(Area)']
    max_list[4] = df.filter(df.District == "翔安").agg({"Area": "max"}).first()['max(Area)']
    max_list[5] = df.filter(df.District == "同安").agg({"Area": "max"}).first()['max(Area)']

    # 返回值是一个list，所以在最后加一个[0]
    mid_list[0] = df.filter(df.District == "海沧").approxQuantile("Area", [0.5], 0.01)[0]
    mid_list[1] = df.filter(df.District == "湖里").approxQuantile("Area", [0.5], 0.01)[0]
    mid_list[2] = df.filter(df.District == "集美").approxQuantile("Area", [0.5], 0.01)[0]
    mid_list[3] = df.filter(df.District == "思明").approxQuantile("Area", [0.5], 0.01)[0]
    mid_list[4] = df.filter(df.District == "翔安").approxQuantile("Area", [0.5], 0.01)[0]
    mid_list[5] = df.filter(df.District == "同安").approxQuantile("Area", [0.5], 0.01)[0]

    all_list = []
    all_list.append(min_list)
    all_list.append(max_list)
    all_list.append(mean_list)
    all_list.append(mid_list)
    return all_list
def spark_analyse_decoration_count(df):
    count_list=[0 for i in range(6)]
    count_list[0] = df.filter(df.District == "海沧").filter(df.Decoration == 1).count()
    count_list[1] = df.filter(df.District == "湖里").filter(df.Decoration == 1).count()
    count_list[2] = df.filter(df.District == "集美").filter(df.Decoration == 1).count()
    count_list[3] = df.filter(df.District == "思明").filter(df.Decoration == 1).count()
    count_list[4] = df.filter(df.District == "翔安").filter(df.Decoration == 1).count()
    count_list[5] = df.filter(df.District == "同安").filter(df.Decoration == 1).count()
    return count_list
def spark_analyse_orientation_count(df):
    count_list=[0 for i in range(6)]
    count_list[0] = df.filter(df.District == "海沧").filter(df.Orientation == 7).count()
    count_list[1] = df.filter(df.District == "湖里").filter(df.Orientation == 7).count()
    count_list[2] = df.filter(df.District == "集美").filter(df.Orientation == 7).count()
    count_list[3] = df.filter(df.District == "思明").filter(df.Orientation == 7).count()
    count_list[4] = df.filter(df.District == "翔安").filter(df.Orientation == 7).count()
    count_list[5] = df.filter(df.District == "同安").filter(df.Orientation == 7).count()
    return count_list

def spark_analyse_District(df):
    print("spark分析,计算统计量")
    # max_list存储各个区的最大值;同理的mean_list, 以及min_list,approxQuantile中位数
    max_list = [0 for i in range(6)]
    mean_list = [1.2 for i in range(6)]
    min_list = [0 for i in range(6)]
    mid_list = [0 for i in range(6)]

    mean_list[0] = df.filter(df.District == "海沧").agg({"Price": "mean"}).first()['avg(Price)']
    mean_list[1] = df.filter(df.District == "湖里").agg({"Price": "mean"}).first()['avg(Price)']
    mean_list[2] = df.filter(df.District == "集美").agg({"Price": "mean"}).first()['avg(Price)']
    mean_list[3] = df.filter(df.District == "思明").agg({"Price": "mean"}).first()['avg(Price)']
    mean_list[4] = df.filter(df.District == "翔安").agg({"Price": "mean"}).first()['avg(Price)']
    mean_list[5] = df.filter(df.District == "同安").agg({"Price": "mean"}).first()['avg(Price)']

    min_list[0] = df.filter(df.District == "海沧").agg({"Price": "min"}).first()['min(Price)']
    min_list[1] = df.filter(df.District == "湖里").agg({"Price": "min"}).first()['min(Price)']
    min_list[2] = df.filter(df.District == "集美").agg({"Price": "min"}).first()['min(Price)']
    min_list[3] = df.filter(df.District == "思明").agg({"Price": "min"}).first()['min(Price)']
    min_list[4] = df.filter(df.District == "翔安").agg({"Price": "min"}).first()['min(Price)']
    min_list[5] = df.filter(df.District == "同安").agg({"Price": "min"}).first()['min(Price)']

    max_list[0] = df.filter(df.District == "海沧").agg({"Price": "max"}).first()['max(Price)']
    max_list[1] = df.filter(df.District == "湖里").agg({"Price": "max"}).first()['max(Price)']
    max_list[2] = df.filter(df.District == "集美").agg({"Price": "max"}).first()['max(Price)']
    max_list[3] = df.filter(df.District == "思明").agg({"Price": "max"}).first()['max(Price)']
    max_list[4] = df.filter(df.District == "翔安").agg({"Price": "max"}).first()['max(Price)']
    max_list[5] = df.filter(df.District == "同安").agg({"Price": "max"}).first()['max(Price)']

    # 返回值是一个list，所以在最后加一个[0]
    print(df.head(5))
    mid_list[0] = df.filter(df.District == "海沧").approxQuantile("Price", [0.5], 0.01)[0]
    mid_list[1] = df.filter(df.District == "湖里").approxQuantile("Price", [0.5], 0.01)[0]
    mid_list[2] = df.filter(df.District == "集美").approxQuantile("Price", [0.5], 0.01)[0]
    mid_list[3] = df.filter(df.District == "思明").approxQuantile("Price", [0.5], 0.01)[0]
    mid_list[4] = df.filter(df.District == "翔安").approxQuantile("Price", [0.5], 0.01)[0]
    mid_list[5] = df.filter(df.District == "同安").approxQuantile("Price", [0.5], 0.01)[0]

    all_list = []
    all_list.append(min_list)
    all_list.append(max_list)
    all_list.append(mean_list)
    all_list.append(mid_list)

    print("结束spark分析")

    return all_list
def spark_linear(df):
    df = df.filter(df.Region == "海沧")
    vec_assmebler = VectorAssembler(inputCols=['Room','Hall','Bathroom','Area','Floor','Total_Floor','Decoration'], outputCol='features')
    features_df = vec_assmebler.transform(df)
    # 结果观察
    features_df.printSchema()
    features_df.show(5, False)
    model_df = features_df.select('features', 'Price2')
    # 数据检视
    model_df.show(5, False)
    train_df, test_df = model_df.randomSplit([0.7, 0.3])
    print((train_df.count(), len(train_df.columns)))
    print((test_df.count(), len(test_df.columns)))
    train_df.describe().show()
    # 构建线性模型
    lin_reg = LinearRegression(labelCol='Price2')
    # 训练集拟合线性模型
    lr_model = lin_reg.fit(train_df)
    # 模型学习到的参数集
    # 1偏置项
    print(lr_model.intercept)
    # 2参数系数向量
    print(lr_model.coefficients)
    print(lr_model.params)
    training_predictions = lr_model.evaluate(train_df)
    # 有效性指标
    # 1 均方误差
    training_predictions.meanSquaredError
    # 2 拟合系数
    print(training_predictions.r2)

    # 测试集上面的效果分析
    test_results = lr_model.evaluate(test_df)
    # 1 均方根误差
    test_results.rootMeanSquaredError
    # 2 均方误差
    test_results.meanSquaredError