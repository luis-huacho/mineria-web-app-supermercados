from django.db import models

# Create your models here.
class Categoria(models.Model):
    id = models.AutoField(primary_key=True)
    idcat = models.IntegerField()
    name = models.CharField(max_length=255)
    market = models.CharField(max_length=255)
    has_children = models.BooleanField(default=False)
    url = models.CharField(max_length=255)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=255)
    meta_tag_description = models.TextField()

    class Meta:
        db_table = 'categories'

    def __str__(self):
        return self.name