## گزارش تمرین عملی سری دوم 
### حسین بابازاده | 401521066

#### Simulated Annealing
- این مکانیزم به طور میانگین چه تاثیری روی الگوریتم میگذارد؟ وجود چنین مکانیزمی در بازی نیاز است؟

حرکت هایی که ما را از گروگان دور میکنند با احتمالی وابسته به دما میتواند کمک کننده باشد. در ابتدا با زیاد بودن دما این مورد بیشتر ممکن است اتفاق بیفتد و ما را از بن بست ها رهایی دهد. اما هر چه جلوتر میرویم احتمال این مورد کمتر میشود که منطقی هم هست چون انتظار میرود در طول زمان به جواب نزدیک تر شویم.


- به نظر شما این آیا استراتژی در این الگوریتم استفاده خواهد شد؟ ممکن است در این الگوریتم در لوپ گیر کنیم؟

در قسمت پیش توضیح دادیم که مکانیزم کاهش تدریجی دما و امکان انجام حرکتی که ما را از گروگان دور می کند چطور کمک میکند. حالا اگر در قدم های جلوتر که دما کاهش پیدا کرده به بن بست بر بخوریم مکانیزم حرکت تصادفی به ما کمک میکند که از بن بست ها رهایی پیدا کنیم.

#### Genetic Algorithm
- به نظر شما پیاده سازی چنین مکانیزمی نیاز است؟ این مکانیزم به طور میانگین باعث 
بهبود الگوریتم میشود؟

بله در واقع اگر crossover به طور درست انجام شود میتواند باعث رهایی از بن بست شود و جهش به درستی اتفاق بیفتد.

- فکر میکنید این مکانیزم به طور میانگین باعث بهبود الگوریتم میشود؟

میتوان گفت که نتیجه به طور میانگین بهتر میشود با توجه به اینکه حرکت تصادقی است چون اگر بن بست داشته باشیم، از آن بن بست بیرون می آییم.


#### Overall
- در کدام یک از این الگوریتم ها احتمال گیر افتادن در بهینه محلی بیشتر است؟

به طور کلی احتمال گیر افتادن در بهینه محلی به این شکل است:
Hill Climbing > Simulated Annealing > Genetic Algorithm


- هر الگوریتم چگونه با این مشکل مقابله میکند؟آیا تپه نوردی با مکانیزم تصادفی به درستی از گیر افتادن در لوپ جلوگیری میکند؟

در واقع میتوان گفت که لزوما از گیر افتادن جلوگیری نمیکند و این فرایند رندوم است و با احتمالی از گیر افتادن جلوگیری میکند.

- کدام الگوریتم در یک محیط پیچیده و پر از مانع عملکرد بهتری دارند؟

من GA رو طوری پیاده سازی کردم که همیشه جواب دارد.
پس جواب من ژنتیک است ولی وابسته به نحوه پیاده سازی است. همچنین بعد از GA الگوریتم Simulated Annealing عملکرد بهتری دارد چون مثل Hill Climbing فقط حرکت رندوم نمیزند و حرکت های اشتباهی که با توجه به احتمالی که دما وابسته است، انجام میدهد، کمک کننده است.

- کدام یک از الگوریتم ها سریع تر به یک نتیجه نهایی می رسد؟

مجدد با توجه به پیاده سازی من GA سریع تر به جواب میرسد چون بعد از تولید نسل ها در انتها حتما جواب داریم در حالی که در Hill Climbing و Simulated Annealing احتمالا نیاز به رندوم ری استارت خواهیم داشت.

- کدام الگوریتم شانس بیشتری برای پیدا کردن بهینه جهانی دارد؟

باز هم GA شانس بیشتری برای این مورد دارد چون Hill Climbing و Simulated Annealing تعداد شروع های تصادفی بیشتری را انجام میدهد، در حالی که در ژنتیک ژن های نسل آخر از نسل های قدیمی به وجود آمده اند و احتمال شروع تصادفی کمتر است.

- در هر الگوریتم، چگونه تنظیمات پارامتر ها بر عملکرد آن تاثیر میگذارد؟ 

Simulated Annealing:‌ در کل اگر دما را با سرعت کمتری پایین بیاوریم احتمالا دیر تر به جواب میرسیم ولی به جواب که میرسیم به global optimum نزدیکتر است.

Genetic Algorithm:‌ اگر اندازه population_size و generation_size بیشتر باشد، مجدد زمان و هزینه بیشتری باید صرف کنیم ولی جواب نهایی به global optimum نزدیک تر خواهد بود.

- مقایسه زمان اجرا و تعداد step های الگوریتم ها

جدول زیر زمان اجرا و تعداد step های ۳ الگوریتم Hill Climbing و Simulated Annealing و Genetic Algorithm را بعد از 20 بار اجرای الگوریتم نشان میدهد.

| Algorithm            | Avg Exec Time (s) | Avg Steps | Avg Obstacle Meets |
|----------------------|-------------------|-----------|--------------------|
| Hill Climbing        | 5.53              | 28.1      | 2.95               |
| Simulated Annealing  | 3.89              | 19.8      | 0.1                |
| Genetic Algorithm    | 3.40              | 14.75     | 0.0                |

توجه شود که تعداد برخورد با مانع در Genetic Algorithm برابر با 0 است چون در انتها برخوردی با موانع ندارد ولی در تولید نسل ها این اتفاق ممکن است بیفتد.

پس همانطور که در بخش های قبل توضیح داده شده به طور کلی Genetic Algorithm برتر از Simulated Annealing و همچنین Simulated Annealing برتر از Hill Climbing است.

البته باز تاکید میکنم که این مقایسه بسیار وابسته به مسئله و نوع پیاده سازی الگوریتم ها هست ولی با توجه به این مسئله ها و پیاده سازی های من مقایسه به شکل گفته شده می باشد.