# 1. ở module generate_estimators thì biến forrest_est nên thành dictionary 
# để lưu estimator trực quan hơn
forest_est.append((f"estimator_{i}", pipe)) # from
forest_est[f'estimator_{i}'] = pipe # to

# từ đó khi mà bạn chạy các estimator sẽ được call ra như thế này
for est in forest.keys():
    estimator = forest[est]
    estimator.fit(X, self.y_)


# ------------------------------------------
# 2. ở module initialise_miss_guess thì bạn nên có validation check cho param method
# check xem value đó có thuộc {"simple", "simple_stratified", "knn", "manual"} không
# bạn có thể ném vào một cái assert ở trên đầu hoặc trong condition thì phần else ném vào raise error


# ------------------------------------------
# 3. các helper function (các function sẽ không sử dụng để call trực tiếp, sử dụng để support các method khác)
# thì nên được đặt tên với dấu "_" ở phía trước
# eg: initialise_miss_guess -> _initialise_miss_guess
# đây là một convention, để bạn và người khác tiện theo dõi code trong 1 object hơn
# biết rõ là method nào sẽ được sử dụng để call trực tiếp


# ------------------------------------------
# 4. tách riêng method fit và transform ra
# bạn có thể thấy thằng sklearn nó hỗ trợ cả 3 method fit, transform và fit_transform đó!
# mình thấy bạn nên tách nó ra để follow principle Separation of Concerns (search GG để tìm hiểu nha hê hê)


# ------------------------------------------
# 5. các class properties (các biến của 1 object định dạng theo kiểu self.x =...)
# thì nên được define ở mục __init__ trước để bạn có thể theo dõi được là có bao nhiêu property trong class
# với các variable sẽ được define ở trong các method sau thì bạn có thể set up nó = None
# rồi sau đó bạn cập nhật lại trong method sau
def __init__():
    self.miss_val_ = None
def define_problem(self,df,target_name,miss_col_name,miss_val=np.nan,ordinal_list=[]):
    self.miss_val_ = miss_val    


