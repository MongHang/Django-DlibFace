var vm = new Vue({
    el: '#app',
    // 修改Vue變數的讀取語法，避免和django範本語法衝突
    delimiters: ['[[', ']]'],
    data: {
        error_name: false,
        error_password: false,
        error_password2: false,
        error_check_password: false,
        error_mobile: false,
        error_allow: false,
        error_name_message: '請輸入5-20個字元的使用者',
        error_password_message: '請輸入8-20位元的密碼',
        error_password2_message: '兩次輸入的密碼不一致',
        error_mobile_message: '請輸入正確的手機號碼',
        error_allow_message: '請勾選用戶協議',
        username: '',
        password: '',
        password2: '',
        mobile: '',
        allow: true
    },
    mounted: function () {
        // 向伺服器獲取圖片驗證碼
        this.generate_image_code();
//        this.error_name_message='我喜歡你'
    },
    methods: {

        // 檢查用戶名
        check_username: function () {
            var re = /^[\u4e00-\u9fa5a-zA-Z0-9_-]{5,20}$/;
            if (re.test(this.username)) {
                this.error_name = false;
            } else {
                this.error_name_message = '請輸入5-20個字元的用戶名';
                this.error_name = true;
            }
         if (!this.error_name) { // 檢查 username
          let url = '/usernames/' + this.username + '/count/';
          axios.get(url).then(response => {
            if (response.data.count == 0) {
              this.error_name = false;
            } else {
              this.error_name = true;
              this.error_name_message = '用戶名已註冊';
            }
          }).catch(error => {
            // 處理錯誤，如果需要的話
          });
        }

        },
        // 檢查密碼
        check_password: function () {
            var re = /^[0-9A-Za-z]{8,20}$/;
            if (re.test(this.password)) {
                this.error_password = false;
            } else {
                this.error_password = true;
            }
        },
        // 確認密碼
        check_password2: function () {
            if (this.password != this.password2) {
                this.error_password2 = true;
            } else {
                this.error_password2 = false;
            }
        },
        // 檢查手機號
        check_mobile: function () {
            var re = /^09\d{8}$/;
            if (re.test(this.mobile)) {
                this.error_mobile = false;
            } else {
                this.error_mobile_message = '您輸入的手機號格式不正確';
                this.error_mobile = true;
            }
            //檢查重複
            if (!this.error_mobile) { // 檢查 mobile
              let url = '/userphones/' + this.mobile + '/count/';
              axios.get(url).then(response => {
                if (response.data.count_phone == 0) {
                  this.error_mobile = false;
                } else {
                  this.error_mobile = true;
                  this.error_mobile_message = '手機號碼已註冊';
                }
              }).catch(error => {
                // 處理錯誤，如果需要的話
              });
            }

        },

        // 檢查是否勾選協議
        check_allow: function () {
            if (!this.allow) {
                this.error_allow = true;
            } else {
                this.error_allow = false;
            }
        },

        // 表單提交
        on_submit(){
            this.check_username();
            this.check_password();
            this.check_password2();
            this.check_mobile();
            // this.check_sms_code();
            this.check_allow();

            if (this.error_name == true || this.error_password == true || this.error_check_password == true
                || this.error_phone == true || this.error_sms_code == true || this.error_allow == true) {
                // 不滿足註冊條件：禁用表單
                window.event.returnValue = false;
                alert("請填入完整資料");
            }
        }
    }
});



